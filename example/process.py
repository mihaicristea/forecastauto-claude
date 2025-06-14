from rembg import remove
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import numpy as np
import os
import cv2

input_path = "input/car.jpg"
output_path = "output/final.jpg"
background_color = (240, 240, 240)  # Studio-style background

def generate_gradient_background(output_path, size=(1920, 1080), color1=(240, 240, 240), color2=(200, 200, 200)):
    """
    Generate a gradient background.
    :param output_path: Path to save the generated background.
    :param size: Size of the background image (width, height).
    :param color1: Starting color of the gradient (RGB tuple).
    :param color2: Ending color of the gradient (RGB tuple).
    """
    width, height = size
    background = Image.new("RGB", size, color1)
    draw = ImageDraw.Draw(background)

    for y in range(height):
        alpha = y / height
        r = int((1 - alpha) * color1[0] + alpha * color2[0])
        g = int((1 - alpha) * color1[1] + alpha * color2[1])
        b = int((1 - alpha) * color1[2] + alpha * color2[2])
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    background.save(output_path)

def generate_background_from_example(example_path, output_path, size=(1920, 1080)):
    """
    Generate a background similar to the example image.
    :param example_path: Path to the example image.
    :param output_path: Path to save the generated background.
    :param size: Size of the background image (width, height).
    """
    example_img = Image.open(example_path).convert("RGB")
    example_img = example_img.resize(size)
    example_img.save(output_path)

def extract_background(example_path, output_path):
    """
    Extract the background from the example image.
    :param example_path: Path to the example image.
    :param output_path: Path to save the extracted background.
    """
    example_img = Image.open(example_path).convert("RGB")
    # Assuming the background is uniform, we can blur the image to smooth out details
    background = example_img.filter(ImageFilter.GaussianBlur(50))
    background.save(output_path)

def extract_logo_and_background(example_path, output_path):
    """
    Extract the logo and background from the example image, removing the car.
    :param example_path: Path to the example image.
    :param output_path: Path to save the extracted logo and background.
    """
    example_img = Image.open(example_path).convert("RGBA")

    # Assuming the car is in the center, we can mask it out
    width, height = example_img.size
    mask = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(mask)

    # Define the bounding box for the car (center area)
    car_box = (width // 4, height // 3, 3 * width // 4, 2 * height // 3)
    draw.rectangle(car_box, fill=0)

    # Apply the mask to remove the car
    background_with_logo = Image.composite(example_img, Image.new("RGBA", example_img.size, (255, 255, 255, 0)), mask)
    background_with_logo = background_with_logo.convert("RGB")
    background_with_logo.save(output_path)

def refine_logo_and_background(example_path, output_path):
    """
    Refine the extraction of logo and background from the example image.
    :param example_path: Path to the example image.
    :param output_path: Path to save the refined logo and background.
    """
    example_img = Image.open(example_path).convert("RGBA")

    # Create a mask to isolate the logo and background
    width, height = example_img.size
    mask = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(mask)

    # Define the bounding box for the car (center area)
    car_box = (width // 4, height // 3, 3 * width // 4, 2 * height // 3)
    draw.rectangle(car_box, fill=0)

    # Blur the mask edges to blend better
    mask = mask.filter(ImageFilter.GaussianBlur(20))

    # Apply the mask to remove the car and refine the background
    refined_background = Image.composite(example_img, Image.new("RGBA", example_img.size, (255, 255, 255, 0)), mask)
    refined_background = refined_background.convert("RGB")
    refined_background.save(output_path)

def enhance_car_quality(car_image):
    """
    Îmbunătățește calitatea imaginii mașinii
    """
    # Îmbunătățire contrast
    enhancer = ImageEnhance.Contrast(car_image)
    car_image = enhancer.enhance(1.2)
    
    # Îmbunătățire claritate
    enhancer = ImageEnhance.Sharpness(car_image)
    car_image = enhancer.enhance(1.3)
    
    # Îmbunătățire saturație
    enhancer = ImageEnhance.Color(car_image)
    car_image = enhancer.enhance(1.1)
    
    return car_image

def add_realistic_shadow(car_image, background):
    """
    Adaugă o umbră realistă sub mașină
    """
    width, height = car_image.size
    
    # Creează o umbră simplă
    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    
    # Desenează umbra în partea de jos
    shadow_y = int(height * 0.85)
    shadow_height = int(height * 0.15)
    
    for i in range(shadow_height):
        alpha = int(50 * (1 - i / shadow_height))  # Umbră care se estompează
        y = shadow_y + i
        if y < height:
            draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
    
    # Blur umbra pentru efect natural
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=8))
    
    return shadow

def blend_with_background(car_image, background):
    """
    Îmbunătățește integrarea mașinii cu fundalul
    """
    # Ajustează marginile pentru integrare mai bună
    alpha = car_image.split()[-1]
    
    # Blur ușor marginile pentru tranziție mai naturală
    alpha_blurred = alpha.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Reconstituie imaginea cu marginile blur
    car_channels = car_image.split()[:3]
    car_integrated = Image.merge("RGBA", car_channels + (alpha_blurred,))
    
    return car_integrated

def color_match(car_image, background):
    """
    Ajustează culorile mașinii pentru a se potrivi cu fundalul
    """
    # Calculează temperatura de culoare a fundalului
    bg_array = np.array(background.convert("RGB"))
    avg_color = np.mean(bg_array, axis=(0, 1))
    
    # Ajustează temperatura culorii mașinii
    car_array = np.array(car_image.convert("RGBA"))
    
    # Factor de ajustare subtil
    adjustment_factor = 0.1
    
    for i in range(3):  # RGB channels
        car_array[:, :, i] = car_array[:, :, i] * (1 - adjustment_factor) + avg_color[i] * adjustment_factor
    
    # Convertește înapoi la imagine
    car_adjusted = Image.fromarray(car_array.astype(np.uint8), "RGBA")
    
    return car_adjusted

def detect_and_replace_license_plate(car_image):
    """
    Detectează și înlocuiește plăcuța de înmatriculare cu text personalizat
    """
    # Convertește imaginea PIL în format OpenCV
    car_cv = cv2.cvtColor(np.array(car_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    # Detectează plăcuțele de înmatriculare folosind Haar Cascade
    # Pentru simplitate, vom folosi detectarea de contururi pentru forme dreptunghiulare
    gray = cv2.cvtColor(car_cv, cv2.COLOR_BGR2GRAY)
    
    # Detectează marginile
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Găsește contururile
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Căută contururi care ar putea fi plăcuțe de înmatriculare
    for contour in contours:
        # Calculează aria și perimetrul
        area = cv2.contourArea(contour)
        if area < 500:  # Plăcuța trebuie să aibă o dimensiune minimă
            continue
            
        # Aproximează conturul la un poligon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Verifică dacă are 4 colțuri (dreptunghi)
        if len(approx) == 4:
            # Calculează aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # Plăcuțele de înmatriculare au un aspect ratio specific (între 2-5)
            if 2.0 <= aspect_ratio <= 5.0 and area > 1000:
                # Înlocuiește zona cu text personalizat
                replace_license_plate_area(car_cv, x, y, w, h)
                break
    
    # Convertește înapoi la PIL
    car_modified = cv2.cvtColor(car_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(car_modified).convert("RGBA")

def replace_license_plate_area(image, x, y, w, h):
    """
    Înlocuiește zona plăcuței cu text personalizat
    """
    # Creează un dreptunghi alb pentru plăcuță
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
    
    # Adaugă textul "FORECASTAUTO"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "FORECASTAUTO"
    
    # Calculează dimensiunea textului pentru a se potrivi în dreptunghi
    font_scale = min(w / 200, h / 40) * 0.8
    thickness = max(1, int(font_scale * 2))
    
    # Calculează poziția centrală pentru text
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    
    # Desenează textul
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

def simple_license_plate_replacement(car_image):
    """
    Înlocuire simplă a unei zone estimate pentru plăcuța de înmatriculare
    """
    # Convertește imaginea la numpy array, păstrând canalul alpha
    car_array = np.array(car_image)
    height, width = car_array.shape[:2]

    # Estimează poziția plăcuței (zona de jos-centru a mașinii)
    plate_width = int(width * 0.25)   # 25% din lățimea imaginii
    plate_height = int(height * 0.08) # 8% din înălțimea imaginii

    # Poziția centrală în partea de jos
    plate_x = int(width * 0.375)   # Centrat orizontal
    plate_y = int(height * 0.75)  # În partea de jos

    # Verifică dacă zona estimată este în limitele imaginii
    if (plate_x + plate_width < width and 
        plate_y + plate_height < height and 
        plate_width > 30 and plate_height > 10):

        # Creează dreptunghiul alb pentru plăcuță
        car_array[plate_y:plate_y+plate_height, plate_x:plate_x+plate_width] = [240, 240, 240, 255]

        # Adaugă margini negre
        border_thickness = 2
        # Marginea de sus și jos
        car_array[plate_y:plate_y+border_thickness, plate_x:plate_x+plate_width] = [20, 20, 20, 255]
        car_array[plate_y+plate_height-border_thickness:plate_y+plate_height, plate_x:plate_x+plate_width] = [20, 20, 20, 255]
        # Marginea stânga și dreapta  
        car_array[plate_y:plate_y+plate_height, plate_x:plate_x+border_thickness] = [20, 20, 20, 255]
        car_array[plate_y:plate_y+plate_height, plate_x+plate_width-border_thickness:plate_x+plate_width] = [20, 20, 20, 255]

        # Convertește la PIL pentru a adăuga text cu font
        car_with_plate = Image.fromarray(car_array.astype(np.uint8))

        # Folosește PIL pentru a adăuga textul
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(car_with_plate)

        # Textul de adăugat
        text = "FORECAST AUTO"

        # Calculează dimensiunea fontului pentru a se potrivi
        font_size = int(plate_height * 0.5)  # 50% din înălțimea plăcuței

        try:
            # Încearcă să folosească un font sistem
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            # Dacă nu găsește fontul, folosește fontul default
            font = ImageFont.load_default()

        # Calculează poziția textului pentru centrare
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = plate_x + (plate_width - text_width) // 2
        text_y = plate_y + (plate_height - text_height) // 2

        # Desenează textul negru
        draw.text((text_x, text_y), text, fill=(0, 0, 0, 255), font=font)

        return car_with_plate

    return car_image

def detect_license_plate_opencv(car_image):
    """
    Detectează automat poziția numărului de înmatriculare folosind OpenCV.
    """
    # Convertește imaginea PIL în format OpenCV
    car_cv = cv2.cvtColor(np.array(car_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    height, width = car_cv.shape[:2]
    
    # Convertește în grayscale
    gray = cv2.cvtColor(car_cv, cv2.COLOR_BGR2GRAY)
    
    # Aplicarea unui filtru bilateral pentru reducerea zgomotului
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Detectarea marginilor
    edges = cv2.Canny(gray, 30, 200, apertureSize=3)
    
    # Găsirea contururilor
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort by area, but process more than just top 10 if needed, or filter more carefully
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20] # Check more contours
    
    license_plate = None
    
    # Criterii mai stricte pentru detect_license_plate_opencv
    min_area = 150  # Redus de la 500 sau 200
    min_w, min_h = 30, 8 # Redus de la 50, 15 sau 40,10
    max_w_ratio, max_h_ratio = 0.30, 0.12  # Max 30% lățime imagine, 12% înălțime
    min_aspect, max_aspect = 2.5, 6.5 # Raport aspect tipic pentru plăcuțe

    # Iterează prin contururi pentru a găsi numărul de înmatriculare
    for contour in contours:
        # Aproximează conturul la un poligon
        epsilon = 0.018 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Verifică dacă are 4 colțuri (dreptunghi)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h) if h > 0 else 0
            area = cv2.contourArea(contour)
            
            # Verifică dacă dimensiunile și aspect ratio-ul corespund unui număr de înmatriculare
            if (min_aspect <= aspect_ratio <= max_aspect and 
                area > min_area and 
                w > min_w and h > min_h and
                w < width * max_w_ratio and h < height * max_h_ratio and
                y > (height * 0.45) and  # Partea de sus a plăcuței trebuie să fie în jumătatea inferioară (mai permisiv, >45%)
                (x + w/2) > (width * 0.10) and (x + w/2) < (width * 0.90) ): # Orizontal centrat (mai permisiv)
                
                # Verificare suplimentară: zona nu este predominant neagră (umbră)
                # Acest lucru poate fi costisitor, dar util
                # plate_roi = gray[y:y+h, x:x+w]
                # if cv2.mean(plate_roi)[0] > 50: # Valoarea medie a pixelilor să fie peste un prag
                
                license_plate = (x, y, w, h)
                print(f"OpenCV Candidate: x={x},y={y},w={w},h={h}, AR={aspect_ratio:.2f}, Area={area}")
                break
    
    # Dacă nu găsește prin contururi, încearcă detectarea prin template matching pentru text
    # Fallback-ul la detect_text_regions este păstrat, deși ar putea necesita și el ajustări
    if license_plate is None:
        print("Detectarea primară OpenCV a eșuat, se încearcă detect_text_regions...")
        license_plate = detect_text_regions(gray) # gray a fost deja calculat
    
    return license_plate

def detect_text_regions(gray_image):
    """
    Detectează regiunile cu text pentru a găsi numărul de înmatriculare.
    """
    height, width = gray_image.shape
    
    # Aplică morfologie pentru a conecta literele
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    
    # Detectarea marginilor
    edges = cv2.Canny(morph, 50, 150)
    
    # Găsirea contururilor
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrează contururile pentru a găsi cele care ar putea fi numere
    potential_plates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        area = w * h
        
        # Criterii pentru numărul de înmatriculare
        if (1.5 <= aspect_ratio <= 8.0 and 
            area > 300 and 
            w > 40 and h > 10 and
            w < width * 0.9 and h < height * 0.4):
            
            potential_plates.append((x, y, w, h, area))
    
    # Sortează după aria și returnează cel mai mare
    if potential_plates:
        potential_plates.sort(key=lambda x: x[4], reverse=True)
        return potential_plates[0][:4]  # returnează doar x, y, w, h
    
    return None

def detect_license_plate_enhanced(car_image):
    """
    Detectare îmbunătățită a numărului de înmatriculare cu multiple metode.
    """
    # Convertește imaginea PIL în format OpenCV
    car_cv = cv2.cvtColor(np.array(car_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    height, width = car_cv.shape[:2]
    
    # Metodă 1: Detectare prin culoare (numerele sunt de obicei albe/galbene)
    hsv = cv2.cvtColor(car_cv, cv2.COLOR_BGR2HSV)
    
    # Mask pentru culori deschise (alb) - mai permisiv
    lower_white = np.array([0, 0, 140])  # Redus min V de la 150 (original 180)
    upper_white = np.array([180, 60, 255]) # Crescut max S de la 50 (original 40)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Mask pentru galben (comun pentru numere EU) - mai permisiv
    lower_yellow = np.array([15, 60, 60]) # Redus min S și V de la 80, 100 (original 100,100)
    upper_yellow = np.array([40, 255, 255]) # Lărgit H de la 35 (original 30)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combină măștile de culoare
    color_mask = cv2.bitwise_or(mask_white, mask_yellow)
    
    # Metodă 2: Detectare prin text folosind gradienți
    gray = cv2.cvtColor(car_cv, cv2.COLOR_BGR2GRAY)
    
    # Aplicare gradient pentru a detecta tranziții de text
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) # ksize=-1 (Scharr) ar putea fi o opțiune
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.uint8(magnitude)
    
    # Combinare cu masca de culoare
    combined = cv2.bitwise_and(magnitude, magnitude, mask=color_mask) 
    
    # Morfologie pentru a conecta caracterele
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))  # Kernel lățit (de la 12,3, original 9,3)
    morphed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.dilate(morphed, None, iterations=2) # Iteratii menținute la 2
    
    # Găsire contururi
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Salvează imaginea de debug
    debug_img = car_cv.copy()
    cv2.imwrite("output/debug_color_mask.jpg", color_mask) # Activat pentru debug
    cv2.imwrite("output/debug_combined.jpg", combined)
    cv2.imwrite("output/debug_morphed_for_contours.jpg", morphed)

    candidates = []
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        area = cv2.contourArea(contour)

        # Criterii INIȚIALE ȘI MAI LENIENTE pentru filtrarea candidaților
        if not (1.5 <= aspect_ratio <= 10.0 and  # Aspect ratio și mai larg (original 2-8)
                area > 100 and  # Aria minimă și mai mică (original 200, inițial 300)
                w > 25 and h > 5 and  # Dimensiuni minime și mai mici (original 40,8 inițial 50,10)
                w < width * 0.6 and h < height * 0.20 and # Dimensiuni maxime puțin mai mari (original 0.5, 0.15)
                y > height * 0.4): # Trebuie să fie în jumătatea de jos, dar mai permisiv (original 0.5)
            continue

        # Calculează soliditatea (cât de plin e conturul față de anvelopa sa convexă)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area > 0 else 0

        # Calculează poziția relativă - numerele sunt FOARTE jos
        bottom_position_score = ((y + h / 2) / height) # Centrul vertical al plăcuței, normalizat
                                                # Scorul e mai mare dacă e mai jos

        # Verifică dacă este într-o zonă tipică pentru numere (centru-jos)
        center_x_contour = x + w/2
        horizontal_center_score = 1.0 - abs(center_x_contour - width/2) / (width/2)  # Cât de centrat e
        
        # Scor pentru aspect ratio ideal (5:1 este ideal pentru numere EU)
        ideal_aspect = 5.0
        aspect_score = 1.0 / (1.0 + abs(aspect_ratio - ideal_aspect) * 2) # Penalizează mai mult deviațiile
        
        # Scor pentru soliditate (ideal > 0.8)
        solidity_score = solidity if solidity > 0.7 else solidity * 0.5 # Penalizează soliditatea mică

        # Bonus/Penalizare pentru poziția verticală
        position_bonus = 1.0
        if y < height * 0.6: # Prea sus
            position_bonus = 0.2
        elif y > height * 0.75: # Poziție bună, jos
            position_bonus = 1.5
        if y + h > height * 0.98: # Atinge marginea de jos, posibil tăiat
            position_bonus *= 0.7

        # Penalizare dacă e prea aproape de marginile laterale (mai puțin probabil să fie număr)
        edge_penalty = 1.0
        if x < width * 0.05 or (x + w) > width * 0.95:
            edge_penalty = 0.5

        # Scor total combinat
        # Ponderi: Aspect Ratio (important), Poziție Orizontală, Soliditate, Poziție Verticală
        total_score = (aspect_score * 0.35 + 
                       horizontal_center_score * 0.25 + 
                       solidity_score * 0.20 + 
                       bottom_position_score * 0.20) * position_bonus * edge_penalty
        
        candidates.append((x, y, w, h, total_score, area, aspect_ratio, solidity))
        
        # Desenează candidatul pe imaginea de debug
        color = (0, 255, 0) if total_score > 0.65 else (0, 165, 255) # Verde pt scoruri bune, Portocaliu pt medii
        if total_score <= 0.5: color = (0,0,255) # Roșu pt scoruri mici

        cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 1)
        cv2.putText(debug_img, f"S:{total_score:.2f} AR:{aspect_ratio:.1f} Sol:{solidity:.2f}", 
                   (x, y-5 if y > 10 else y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Salvează imaginea de debug (convertită la RGB dacă e necesar pentru vizualizare)
    # Asigură-te că folderul 'output' există
    if not os.path.exists("output"):
        os.makedirs("output")
    cv2.imwrite("output/debug_detection.jpg", debug_img)
    print(f"Detectate {len(candidates)} candidați potențiali pentru numărul de înmatriculare")
    
    if candidates:
        # Sortează după scor și returnează cel mai bun
        candidates.sort(key=lambda x: x[4], reverse=True)
        best_candidate = candidates[0]
        print(f"Cel mai bun candidat: x={best_candidate[0]}, y={best_candidate[1]}, w={best_candidate[2]}, h={best_candidate[3]}, score={best_candidate[4]:.2f}, aspect_ratio={best_candidate[6]:.2f}, solidity={best_candidate[7]:.2f}")
        
        # Verifică dacă scorul este suficient de mare
        # Am redus pragul, deoarece sistemul de scoring este mai complex
        if best_candidate[4] > 0.6:  # Prag de acceptare ajustat
            return best_candidate[:4]
        else:
            print(f"Scorul cel mai bun ({best_candidate[4]:.2f}) este prea mic. Se încearcă alte metode.")
    
    return None

def simple_license_plate_detection(car_image):
    """
    Detectare simplificată a numărului de înmatriculare bazată pe caracteristici vizuale.
    """
    print("Încearcă detectarea simplificată...")
    
    # Convertește imaginea PIL în format OpenCV
    car_cv = cv2.cvtColor(np.array(car_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    height, width = car_cv.shape[:2]
    
    # Convertește în grayscale
    gray = cv2.cvtColor(car_cv, cv2.COLOR_BGR2GRAY)
    
    # Aplicare threshold pentru a obține zone albe/deschise (potențiale numere)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Găsire contururi
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Salvează imaginea de debug
    debug_img = car_cv.copy()
    cv2.imwrite("output/debug_threshold.jpg", thresh)
    
    best_candidates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        area = cv2.contourArea(contour)
        
        # Criterii simple pentru numărul de înmatriculare
        if (3.0 <= aspect_ratio <= 8.0 and  # Aspect ratio tipic pentru numere
            area > 200 and  # Aria minimă
            w > 60 and h > 12 and  # Dimensiuni minime rezonabile
            w < width * 0.4 and h < height * 0.1 and  # Dimensiuni maxime
            y > height * 0.6 and  # În partea de jos
            x > width * 0.1 and x + w < width * 0.9):  # Nu la marginile extreme
            
            # Calculează scorul simplu bazat pe poziție și aspect ratio
            center_x = x + w/2
            horizontal_center_score = 1.0 - abs(center_x - width/2) / (width/2)
            vertical_position_score = (y + h/2) / height  # Mai jos = scor mai mare
            aspect_score = 1.0 / (1.0 + abs(aspect_ratio - 5.0))  # 5:1 este ideal
            
            total_score = (horizontal_center_score * 0.4 + 
                          vertical_position_score * 0.3 + 
                          aspect_score * 0.3)
            
            best_candidates.append((x, y, w, h, total_score, area, aspect_ratio))
            
            # Desenează candidatul
            color = (0, 255, 0) if total_score > 0.6 else (0, 0, 255)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(debug_img, f"S:{total_score:.2f}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imwrite("output/debug_simple_detection.jpg", debug_img)
    print(f"Detectare simplă: găsite {len(best_candidates)} candidați")
    
    if best_candidates:
        # Sortează după scor și returnează cel mai bun
        best_candidates.sort(key=lambda x: x[4], reverse=True)
        best = best_candidates[0]
        print(f"Cel mai bun candidat simplu: x={best[0]}, y={best[1]}, w={best[2]}, h={best[3]}, score={best[4]:.2f}")
        
        if best[4] > 0.5:  # Prag mai mic pentru metoda simplă
            return best[:4]
    
    return None

def fallback_license_plate_replacement(car_image, logo_path):
    """
    Fallback pentru înlocuirea numărului de înmatriculare folosind estimare pe baza dimensiunilor mașinii.
    """
    print("Se folosește metoda fallback pentru înlocuirea numărului de înmatriculare...")
    
    # Convertește imaginea la numpy array pentru analiză
    car_array = np.array(car_image)
    height, width = car_array.shape[:2]

    # Estimează poziția plăcuței (zona de jos-centru a mașinii)
    plate_width = int(width * 0.20)   # 20% din lățimea imaginii
    plate_height = int(height * 0.06) # 6% din înălțimea imaginii

    # Poziția centrală în partea de jos
    plate_x = int(width * 0.40)   # Centrat orizontal
    plate_y = int(height * 0.80)  # În partea de jos

    # Verifică dacă zona estimată este în limitele imaginii
    if (plate_x + plate_width < width and 
        plate_y + plate_height < height and 
        plate_width > 30 and plate_height > 10):

        print(f"Fallback: Înlocuiește zona estimată la x={plate_x}, y={plate_y}, w={plate_width}, h={plate_height}")
        
        # Creează dreptunghiul alb și adaugă logo-ul
        car_with_plate = car_image.copy()
        draw = ImageDraw.Draw(car_with_plate)
        
        # Desenează dreptunghiul alb cu bordură neagră
        draw.rectangle([plate_x, plate_y, plate_x + plate_width, plate_y + plate_height], 
                       fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=2)
        
        # Adaugă padding pentru logo
        logo_padding = 4
        logo_x = plate_x + logo_padding
        logo_y = plate_y + logo_padding
        logo_w = plate_width - (logo_padding * 2)
        logo_h = plate_height - (logo_padding * 2)
        
        # Încarcă și redimensionează logo-ul
        logo = Image.open(logo_path).convert("RGBA")
        logo = logo.resize((logo_w, logo_h), Image.LANCZOS)
        car_with_plate.paste(logo, (logo_x, logo_y), logo)
        
        return car_with_plate

    print("Fallback: Nu s-a putut estima o zonă validă pentru numărul de înmatriculare.")
    return car_image

def simple_license_plate_replacement_with_logo(car_image, logo_path):
    """
    Detectează automat și înlocuiește numărul de înmatriculare cu logo-ul, încadrând logo-ul perfect pe plăcuță.
    """
    print("Începe detectarea numărului de înmatriculare...")
    
    # Încearcă mai întâi detectarea simplă
    plate_coords = simple_license_plate_detection(car_image)
    
    if plate_coords is None:
        print("Detectarea simplă a eșuat. Încearcă detectarea îmbunătățită...")
        plate_coords = detect_license_plate_enhanced(car_image)
    
    if plate_coords is None:
        print("Detectarea îmbunătățită a eșuat. Încearcă detectarea OpenCV...")
        plate_coords = detect_license_plate_opencv(car_image)
    
    if plate_coords is None:
        print("Toate metodele de detectare au eșuat. Se folosește estimarea implicită.")
        return fallback_license_plate_replacement(car_image, logo_path)
    
    plate_x, plate_y, plate_w, plate_h = plate_coords
    
    # Extinde zona pentru acoperire completă
    padding = int(plate_h * 0.10)  # 10% padding proporțional
    plate_x = max(0, plate_x - padding)
    plate_y = max(0, plate_y - padding)
    plate_w = min(car_image.width - plate_x, plate_w + 2 * padding)
    plate_h = min(car_image.height - plate_y, plate_h + 2 * padding)

    print(f"Numărul de înmatriculare detectat la: x={plate_x}, y={plate_y}, w={plate_w}, h={plate_h}")
    
    # Creează dreptunghiul alb cu bordură neagră
    car_with_plate = car_image.copy()
    draw = ImageDraw.Draw(car_with_plate)
    draw.rectangle([plate_x, plate_y, plate_x + plate_w, plate_y + plate_h], 
                   fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=2)

    # Încarcă logo-ul și păstrează raportul de aspect
    logo = Image.open(logo_path).convert("RGBA")
    logo_aspect = logo.width / logo.height
    plate_aspect = plate_w / plate_h

    # Calculează dimensiunea maximă a logo-ului care încape în plăcuță
    if logo_aspect > plate_aspect:
        # Logo-ul e mai lat decât plăcuța: limitează după lățime
        logo_w = plate_w - 2 * padding
        logo_h = int(logo_w / logo_aspect)
        if logo_h > plate_h - 2 * padding:
            logo_h = plate_h - 2 * padding
            logo_w = int(logo_h * logo_aspect)
    else:
        # Logo-ul e mai înalt: limitează după înălțime
        logo_h = plate_h - 2 * padding
        logo_w = int(logo_h * logo_aspect)
        if logo_w > plate_w - 2 * padding:
            logo_w = plate_w - 2 * padding
            logo_h = int(logo_w / logo_aspect)

    # Centrează logo-ul pe plăcuță
    logo_x = plate_x + (plate_w - logo_w) // 2
    logo_y = plate_y + (plate_h - logo_h) // 2
    logo = logo.resize((logo_w, logo_h), Image.LANCZOS)
    car_with_plate.paste(logo, (logo_x, logo_y), logo)

    # Salvează o versiune cu conturul detectat pentru debugging
    debug_final = car_with_plate.copy().convert("RGB")
    debug_draw = ImageDraw.Draw(debug_final)
    debug_draw.rectangle([plate_x, plate_y, plate_x + plate_w, plate_y + plate_h], 
                        outline=(255, 0, 0), width=3)
    debug_final.save("output/debug_final.jpg")

    return car_with_plate

# Eliminare fundal din car.jpg
with open(input_path, "rb") as f:
    input_img = f.read()
    output_img = remove(input_img)

# Salvează mașina fără fundal (cu transparență)
with open("output/car_no_bg.png", "wb") as f:
    f.write(output_img)

# Încarcă mașina fără fundal și fundalul din background.jpg
car = Image.open("output/car_no_bg.png").convert("RGBA")
background = Image.open("background.jpg").convert("RGBA")

# Redimensionează fundalul să se potrivească cu dimensiunea mașinii
background = background.resize(car.size)

# Modifică plăcuța de înmatriculare cu logo-ul "logo-plate.png"
car = simple_license_plate_replacement_with_logo(car, "logo-plate.png")

# Îmbunătățește calitatea mașinii
car = enhance_car_quality(car)

# Adaugă umbră realistă
shadow = add_realistic_shadow(car, background)
background = Image.alpha_composite(background, shadow)

# Potrivește culorile mașinii cu fundalul
car = color_match(car, background)

# Combină mașina cu noul fundal
final_image = Image.alpha_composite(background, car)
final_image = final_image.convert("RGB")
final_image.save(output_path)
