from smolagents import Tool, CodeAgent, TransformersModel
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, SegformerForSemanticSegmentation, AutoImageProcessor
import numpy as np
import os
from geoclip import GeoCLIP
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import reverse_geocoder as rg
import socket
import torch.multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import time
import threading


# ==== STEP 1: Setup CLIP ====
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained(
    "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/clip-vit-base-patch32"
).to(device)
clip_processor = CLIPProcessor.from_pretrained(
    "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/clip-vit-base-patch32"
)

streetclip_model = CLIPModel.from_pretrained(
    "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/StreetCLIP"
)
streetclip_processor = CLIPProcessor.from_pretrained(
    "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/StreetCLIP"
)

model_name = "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/segformer"
feature_extractor = AutoImageProcessor.from_pretrained(model_name)
segformer_model = SegformerForSemanticSegmentation.from_pretrained(model_name)
segformer_model.to(device)

# Mappa id → label
id2label = segformer_model.config.id2label


# ==== GPU MODEL MANAGER - Mantiene i modelli in VRAM ====
class GPUModelManager:
    """Singleton per gestire modelli GPU persistenti"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.models = {}
            self._lock = threading.Lock()
            self._initialized = True
            print(f"🚀 GPU Model Manager inizializzato su device: {self.device}")
    
    def load_streetclip(self):
        """Carica StreetCLIP una sola volta e lo mantiene in VRAM"""
        with self._lock:
            if 'streetclip' not in self.models:
                print("📥 Caricamento StreetCLIP in VRAM...")
                self.models['streetclip'] = {
                    'model': CLIPModel.from_pretrained(
                        "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/StreetCLIP"
                    ).to(self.device),
                    'processor': CLIPProcessor.from_pretrained(
                        "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/StreetCLIP"
                    )
                }
                print("✅ StreetCLIP caricato e pronto in VRAM")
            return self.models['streetclip']
    
    def load_geoclip(self):
        """Carica GeoCLIP una sola volta e lo mantiene in VRAM"""
        with self._lock:
            if 'geoclip' not in self.models:
                print("📥 Caricamento GeoCLIP in VRAM...")
                self.models['geoclip'] = GeoCLIP().to(self.device)
                print("✅ GeoCLIP caricato e pronto in VRAM")
            return self.models['geoclip']
    
    def get_streetclip(self):
        """Ottieni StreetCLIP (caricandolo se necessario)"""
        if 'streetclip' not in self.models:
            return self.load_streetclip()
        return self.models['streetclip']
    
    def get_geoclip(self):
        """Ottieni GeoCLIP (caricandolo se necessario)"""
        if 'geoclip' not in self.models:
            return self.load_geoclip()
        return self.models['geoclip']
    
    def is_model_loaded(self, model_name):
        """Verifica se un modello è già caricato"""
        return model_name in self.models
    
    def get_memory_usage(self):
        """Ottieni utilizzo memoria GPU"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                'free': (torch.cuda.get_device_properties(0).total_memory - 
                        torch.cuda.memory_reserved()) / 1024**3        # GB
            }
        return None

# Inizializza il manager globale
gpu_manager = GPUModelManager()

# Geopy per conversione coordinate <-> stato
geolocator = Nominatim(user_agent="LocalizatorGeoAgent")

# Funzione per controllo internet
def internet_available():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

# Lista label StreetCLIP
labels = [
    "Albania", "Andorra", "Argentina", "Australia", "Austria", "Bangladesh", "Belgium",
    "Bermuda", "Bhutan", "Bolivia", "Botswana", "Brazil", "Bulgaria", "Cambodia", "Canada",
    "Chile", "China", "Colombia", "Croatia", "Czech Republic", "Denmark", "Dominican Republic",
    "Ecuador", "Estonia", "Finland", "France", "Germany", "Ghana", "Greece", "Greenland", "Guam",
    "Guatemala", "Hungary", "Iceland", "India", "Indonesia", "Ireland", "Israel", "Italy", "Japan",
    "Jordan", "Kenya", "Kyrgyzstan", "Laos", "Latvia", "Lesotho", "Lithuania", "Luxembourg",
    "Macedonia", "Madagascar", "Malaysia", "Malta", "Mexico", "Monaco", "Mongolia", "Montenegro",
    "Netherlands", "New Zealand", "Nigeria", "Norway", "Pakistan", "Palestine", "Peru", "Philippines",
    "Poland", "Portugal", "Puerto Rico", "Romania", "Russia", "Rwanda", "Senegal", "Serbia", "Singapore",
    "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sri Lanka", "Swaziland", "Sweden",
    "Switzerland", "Taiwan", "Thailand", "Tunisia", "Turkey", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "United States", "Uruguay"
]



# ==== STEP 2: Database in memoria ====
image_db = []  # Lista di dict: {path, embedding, text}


def embed_image(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb.cpu().numpy().flatten()


# ==== STEP 3: Download immagini monumenti ====
monuments = [
    (
        "big_ben.png",
        "Big Ben",
        "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/test.jpg",
    ),
    (
        "statue_liberty.png",
        "Statua della Libertà",
        "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/statua_liberta.jpg",
    ),
    (
        "tour_eiffel.png",
        "Tour Eiffel",
        "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/tour_eiffel.jpg",
    ),
    (
        "colosseo.png",
        "Colosseo",
        "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/colosseo.jpg",
    ),
    (
        "cristo_redentor.png",
        "Cristo Redentore",
        "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/cristo_redentor.jpg",
    ),
]

for filename, name, url in monuments:
    img = Image.open(url).convert("RGB")
    # img.save(filename)
    emb = embed_image(img)
    image_db.append({"path": filename, "embedding": emb, "text": name})


# ==== STEP 4: Definizione Tool ====
class ImageRetrievalTool(Tool):
    name = "image_retrieval"
    description = """Tool for extract information, name and description from image."""
    inputs = {
        "image_path": {"type": "string", "description": "Path of the image to analyze and retrieve information."}
    }
    output_type = "string"

    def forward(self, image_path: str) -> str:
        output_dir = "tmp_segments"
        os.makedirs(output_dir, exist_ok=True)
        segments_info = self.segment_image_and_get_paths(image_path, output_dir, conf_threshold=0.6)

        results_str = []
        for seg_path, label_text, seg_conf in segments_info:
            query_img = Image.open(seg_path).convert("RGB")
            query_emb = embed_image(query_img)

            sims = []
            for entry in image_db:
                sim = np.dot(query_emb, entry["embedding"]) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(entry["embedding"])
                )
                if sim >= 0.6:
                    sims.append((sim, entry["text"]))

            sims.sort(reverse=True, key=lambda x: x[0])

            # Crea descrizione dettagliata per questo segmento
            match_desc = (
                ", ".join([f"{name} ({score:.2f})" for score, name in sims])
                if sims else "No matches above similarity threshold"
            )

            segment_report = (
                f"Segment Path: {seg_path}\n"
                f"Description: {label_text}\n"
                f"Confidence: {seg_conf:.2f}\n"
                f"Matches: {match_desc}\n"
                "----------------------------------------"
            )
            results_str.append(segment_report)

        return "\n".join(results_str)

    def segment_image_and_get_paths(self, image_path, output_dir="output_segments", conf_threshold=0.5):
        os.makedirs(output_dir, exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size

        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = segformer_model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=(orig_height, orig_width), mode="bilinear", align_corners=False
        )
        probs = torch.nn.functional.softmax(upsampled_logits, dim=1)[0]
        seg = torch.argmax(probs, dim=0).cpu().numpy()
        unique_segments = np.unique(seg)
        unique_segments = unique_segments[unique_segments != 0]

        def get_position_name(mask):
            ys, xs = np.where(mask == 1)
            if len(xs) == 0 or len(ys) == 0:
                return "unknown"
            cx, cy = xs.mean(), ys.mean()
            if cx < orig_width / 3:
                horizontal_pos = "left"
            elif cx < 2 * orig_width / 3:
                horizontal_pos = "center"
            else:
                horizontal_pos = "right"
            if cy < orig_height / 3:
                vertical_pos = "top"
            elif cy < 2 * orig_height / 3:
                vertical_pos = "middle"
            else:
                vertical_pos = "bottom"
            return f"{vertical_pos} {horizontal_pos}" if vertical_pos != "middle" else horizontal_pos

        segment_paths = []
        for seg_id in unique_segments:
            mask = (seg == seg_id).astype(np.uint8)
            seg_conf = probs[seg_id][mask == 1].mean().item()
            if seg_conf < conf_threshold:
                continue

            position_name = get_position_name(mask)
            class_name = id2label.get(int(seg_id), f"class_{seg_id}")
            label_text = f"{class_name} - {position_name}"

            img_np = np.array(image)
            alpha = (mask * 255).astype(np.uint8)
            rgba = np.dstack((img_np, alpha))
            seg_img = Image.fromarray(rgba, mode="RGBA")

            seg_path = os.path.join(output_dir, f"{class_name}_{position_name}.png")
            seg_img.save(seg_path)
            segment_paths.append((seg_path, label_text, seg_conf))

        return segment_paths


class Localizator(Tool):
    name = "localizator"
    description = """Comprehensive geolocation tool that determines the geographical location of images."""
    inputs = {
        "image_path": {"type": "string", "description": "Path of the image to geolocate."}
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        # Inizializza multiprocessing per CUDA
        mp.set_start_method('spawn', force=True)
        
        # Pre-carica i modelli all'inizializzazione se possibile
        self._preload_models()

    def _preload_models(self):
        """Pre-carica i modelli in VRAM per prestazioni ottimali"""
        print("🔄 Pre-caricamento modelli in corso...")
        
        # Carica sempre StreetCLIP
        gpu_manager.load_streetclip()
        
        # Carica GeoCLIP solo se c'è internet
        if internet_available():
            try:
                gpu_manager.load_geoclip()
            except Exception as e:
                print(f"⚠️ Errore nel caricamento GeoCLIP: {e}")
        
        # Mostra utilizzo memoria
        memory_info = gpu_manager.get_memory_usage()
        if memory_info:
            print(f"💾 Memoria GPU - Allocata: {memory_info['allocated']:.2f}GB, "
                  f"Riservata: {memory_info['reserved']:.2f}GB, "
                  f"Libera: {memory_info['free']:.2f}GB")

    def streetclip_worker(self, image_path, result_queue):
        """Worker process per StreetCLIP con modelli persistenti in VRAM"""
        try:
            # Usa i modelli già caricati dal manager
            streetclip_models = gpu_manager.get_streetclip()
            streetclip_model_worker = streetclip_models['model']
            streetclip_processor_worker = streetclip_models['processor']
            
            device = streetclip_model_worker.device
            
            img = Image.open(image_path).convert("RGB")
            street_inputs = streetclip_processor_worker(
                text=labels, images=img, return_tensors="pt", padding=True
            ).to(device)
            
            with torch.no_grad():
                street_outputs = streetclip_model_worker(**street_inputs)
            
            logits_per_image = street_outputs.logits_per_image
            prediction = logits_per_image.softmax(dim=1)
            sorted_confidences = sorted(
                {labels[i]: float(prediction[0][i].item()) for i in range(len(labels))}.items(),
                key=lambda item: item[1], reverse=True
            )
            street_country, street_conf = sorted_confidences[0]
            
            result_queue.put(('streetclip', street_country, street_conf))
            print(f"✅ StreetCLIP completato: {street_country} ({street_conf*100:.2f}%)")
            
        except Exception as e:
            print(f"❌ Errore StreetCLIP: {e}")
            result_queue.put(('streetclip_error', str(e), 0.0))

    def geoclip_worker(self, image_path, result_queue):
        """Worker process per GeoCLIP con modelli persistenti in VRAM"""
        try:
            # Usa il modello già caricato dal manager
            geoclip_model_worker = gpu_manager.get_geoclip()
            
            top_pred_gps, _ = geoclip_model_worker.predict(image_path, top_k=1)
            geo_lat, geo_lon = top_pred_gps[0]
            
            result_queue.put(('geoclip', geo_lat, geo_lon))
            print(f"✅ GeoCLIP completato: {geo_lat:.6f}, {geo_lon:.6f}")
            
        except Exception as e:
            print(f"❌ Errore GeoCLIP: {e}")
            result_queue.put(('geoclip_error', str(e), 0.0, 0.0))

    def forward(self, image_path: str) -> str:
        print(f"🔍 Avvio localizzazione per: {image_path}")
        
        # Controlla la connessione internet
        has_internet = internet_available()
        print(f"🌐 Connessione internet: {'✅ Disponibile' if has_internet else '❌ Non disponibile'}")
        
        # Mostra stato dei modelli
        streetclip_loaded = gpu_manager.is_model_loaded('streetclip')
        geoclip_loaded = gpu_manager.is_model_loaded('geoclip')
        print(f"🧠 Modelli caricati - StreetCLIP: {'✅' if streetclip_loaded else '❌'}, "
              f"GeoCLIP: {'✅' if geoclip_loaded else '❌'}")
        
        # Crea code per i risultati dei processi
        result_queue = Queue()
        processes = []
        
        # Avvia sempre StreetCLIP
        print("🚀 Avvio processo StreetCLIP...")
        streetclip_process = Process(
            target=self.streetclip_worker,
            args=(image_path, result_queue)
        )
        streetclip_process.start()
        processes.append(streetclip_process)
        
        # Avvia GeoCLIP solo se c'è internet e il modello è disponibile
        geoclip_process = None
        if has_internet and geoclip_loaded:
            print("🚀 Avvio processo GeoCLIP...")
            geoclip_process = Process(
                target=self.geoclip_worker,
                args=(image_path, result_queue)
            )
            geoclip_process.start()
            processes.append(geoclip_process)
        
        # Raccogli i risultati
        results = {}
        expected_results = 2 if (has_internet and geoclip_loaded) else 1
        timeout = 500  # timeout aumentato per modelli pesanti
        start_time = time.time()
        
        print(f"⏳ Attesa risultati ({expected_results} processi, timeout: {timeout}s)...")
        
        while len(results) < expected_results and (time.time() - start_time) < timeout:
            if not result_queue.empty():
                result = result_queue.get()
                if result[0] == 'streetclip':
                    results['streetclip'] = (result[1], result[2])  # country, confidence
                elif result[0] == 'geoclip':
                    results['geoclip'] = (result[1], result[2])  # lat, lon
                elif result[0] == 'streetclip_error':
                    results['streetclip_error'] = result[1]
                elif result[0] == 'geoclip_error':
                    results['geoclip_error'] = result[1]
            time.sleep(0.1)  # Breve pausa per evitare busy waiting
        
        # Termina tutti i processi
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=500)
        
        print("🏁 Elaborazione completata, generazione risultati...")
        
        # Costruisci il risultato finale
        result_text = "🌍 **Risultati Localizzazione** 🌍\n"
        
        # Risultati StreetCLIP
        if 'streetclip' in results:
            street_country, street_conf = results['streetclip']
            street_lat, street_lon = self.get_country_coordinates(street_country, has_internet)
            street_city, street_country_name = self.reverse_geocode(street_lat, street_lon, has_internet)
            
            result_text += (
                f"📌 **StreetCLIP**: {street_city}, {street_country_name} ({street_conf*100:.2f}%)\n"
                f"   Lat: {street_lat:.6f}, Lon: {street_lon:.6f}\n"
            )
        elif 'streetclip_error' in results:
            result_text += f"⚠️ **StreetCLIP**: Errore - {results['streetclip_error']}\n"
        else:
            result_text += "⚠️ **StreetCLIP**: Timeout o processo interrotto\n"
        
        # Risultati GeoCLIP
        if has_internet and geoclip_loaded:
            if 'geoclip' in results:
                geo_lat, geo_lon = results['geoclip']
                geo_city, geo_country = self.reverse_geocode(geo_lat, geo_lon, has_internet)
                
                result_text += (
                    f"\n📌 **GeoCLIP**: {geo_city}, {geo_country}\n"
                    f"   Lat: {geo_lat:.6f}, Lon: {geo_lon:.6f}"
                )
            elif 'geoclip_error' in results:
                result_text += f"\n⚠️ **GeoCLIP**: Errore - {results['geoclip_error']}"
            else:
                result_text += "\n⚠️ **GeoCLIP**: Timeout o processo interrotto"
        else:
            if not has_internet:
                result_text += "\n⚠️ **GeoCLIP**: Non disponibile (nessuna connessione internet)"
            elif not geoclip_loaded:
                result_text += "\n⚠️ **GeoCLIP**: Modello non caricato in VRAM"

        return result_text

    def internet_available(self):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False

    def get_country_coordinates(self, country_name, has_internet=None):
        if has_internet is None:
            has_internet = self.internet_available()
            
        if has_internet:
            try:
                location = geolocator.geocode(country_name, timeout=500)
                if location:
                    return location.latitude, location.longitude
            except GeocoderTimedOut:
                pass
                
        # FALLBACK OFFLINE - usa reverse_geocoder per ottenere coordinate approssimative
        try:
            # Cerca nel database di reverse_geocoder per trovare una città del paese
            # Nota: questo è un workaround poiché reverse_geocoder è principalmente per reverse geocoding
            # Potresti voler aggiungere un database locale di coordinate dei paesi per maggiore accuratezza
            all_locations = rg.search((0.0, 0.0))  # Questo non è ottimale, ma è un esempio
            
            # Prova a trovare una corrispondenza basata sui codici paese
            country_code_map = {
                'United States': 'US', 'United Kingdom': 'GB', 'France': 'FR', 'Germany': 'DE',
                'Italy': 'IT', 'Spain': 'ES', 'Japan': 'JP', 'China': 'CN', 'Brazil': 'BR',
                'Canada': 'CA', 'Australia': 'AU', 'Russia': 'RU', 'India': 'IN'
                # Aggiungi altri mapping se necessario
            }
            
            target_cc = country_code_map.get(country_name, country_name[:2].upper())
            
            # Database semplificato di coordinate centrali per alcuni paesi principali
            country_coords = {
                'US': (39.8283, -98.5795), 'GB': (55.3781, -3.4360), 'FR': (46.6034, 2.2137),
                'DE': (51.1657, 10.4515), 'IT': (41.8719, 12.5674), 'ES': (40.4637, -3.7492),
                'JP': (36.2048, 138.2529), 'CN': (35.8617, 104.1954), 'BR': (-14.2350, -51.9253),
                'CA': (56.1304, -106.3468), 'AU': (-25.2744, 133.7751), 'RU': (61.5240, 105.3188),
                'IN': (20.5937, 78.9629)
            }
            
            if target_cc in country_coords:
                return country_coords[target_cc]
                
        except Exception as e:
            print(f"Errore nel fallback offline: {e}")
            
        return (0.0, 0.0)  # Coordinate di default se tutto fallisce

    def reverse_geocode(self, lat, lon, has_internet=None):
        if has_internet is None:
            has_internet = self.internet_available()
            
        if has_internet:
            try:
                location = geolocator.reverse((lat, lon), exactly_one=True, timeout=500)
                if location and location.raw.get("address"):
                    city = location.raw["address"].get("city") or location.raw["address"].get("town") or location.raw["address"].get("village") or "Unknown"
                    country = location.raw["address"].get("country", "Unknown")
                    return city, country
            except GeocoderTimedOut:
                pass

        # FALLBACK OFFLINE con reverse_geocoder
        try:
            results = rg.search((lat, lon))
            if results:
                entry = results[0]
                return entry["name"], entry["cc"]
        except Exception as e:
            print(f"Errore nel reverse geocoding offline: {e}")
            
        return "Unknown", "Unknown"

# ==== STEP 5: Agente smolagents ====
retrieval_tool = ImageRetrievalTool()
localizator = Localizator()
# agent_model = TransformersModel(model_id="/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/TinyAgent-1.1B", trust_remote_code=True, device_map="auto")
agent_model = TransformersModel(
    model_id="/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/Qwen2.5-Coder-7B-Instruct",
    trust_remote_code=True,
    device_map="auto",
)

memory_info = gpu_manager.get_memory_usage()
if memory_info:
    print(f"💾 Memoria GPU - Allocata: {memory_info['allocated']:.2f}GB, "
            f"Riservata: {memory_info['reserved']:.2f}GB, "
            f"Libera: {memory_info['free']:.2f}GB")
    
    
# agent_model = TransformersModel(model_id="/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/Qwen3-4B-Thinking-2507", trust_remote_code=True, device_map="auto")
agent = CodeAgent(
    tools=[localizator, retrieval_tool],
    model=agent_model,
    additional_authorized_imports=["PIL", "torch", "transformers", "numpy", "io"],
    planning_interval=3,  # Esegui il planning ogni 3 step
)

# ==== STEP 6: Test con domanda più dettagliata ====
file_path = "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data/img_segm/building_center.png"

# Domanda più dettagliata e specifica
detailed_query = f"Localize the image: {file_path} and then give some info about it."

result = agent.run(detailed_query)
print(result)