import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import os

largest_exist = False
center_x, center_y, radius = None, None, None
bright_image = False
segmented_region_display = False
combined_image = False

def load_and_threshold(image_path):
    global bright_image
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    bright_image = np.mean(gray_image) > 78

    def dull_image(image, factor=0.32):
        return np.clip(image * factor, 0, 255).astype(np.uint8)

    if bright_image:
        image = dull_image(gray_image)
    else:
        image = gray_image

    _, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return image, thresh_image

def find_contours(thresh_image):
    contours, _ = cv2.findContours(thresh_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def calculate_areas_and_bounding_boxes(contours):
    areas = []
    bounding_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, x + w, y + h))
    return areas, bounding_boxes

def select_largest_object(areas, bounding_boxes):
    max_area = 0
    selected_object = None
    for i, (area, (min_x, min_y, max_x, max_y)) in enumerate(zip(areas, bounding_boxes)):
        width = max_x - min_x
        height = max_y - min_y
        
        if area > max_area and width > 300 and height > 300:
            max_area = area
            selected_object = (i + 1, area, (min_x, min_y, max_x, max_y))
    return selected_object

def create_mask_and_apply(image, selected_object):
    global largest_exist
    global center_x
    global center_y
    global radius
    output_image = np.zeros_like(image)
    
    if selected_object:
        largest_exist = True
        obj_index, obj_area, (min_x, min_y, max_x, max_y) = selected_object
        # print(f"Selected Object {obj_index}: Area = {obj_area:.2f} pixels, Bounding Box = ({min_x}, {min_y}) to ({max_x}, {max_y})")
        
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        radius = min((max_x - min_x) // 2, (max_y - min_y) // 2)

        mask = np.zeros_like(image)
        cv2.circle(mask, (center_x, center_y), radius, (255), thickness=-1)

        masked_region = cv2.bitwise_and(image, mask)
        _, otsu_thresh = cv2.threshold(masked_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        output_image[masked_region > 0] = otsu_thresh[masked_region > 0]

        points_covered = np.argwhere(mask == 255)
        num_points = len(points_covered)
        
        min_x_mask, min_y_mask = np.min(points_covered, axis=0)
        max_x_mask, max_y_mask = np.max(points_covered, axis=0)

        # print(f"Points covered by the circular mask: {num_points}")
        # print(f"Range of points covered by the mask: min({min_x_mask}, {min_y_mask}) to max({max_x_mask}, {max_y_mask})")

    else:
        largest_exist = False
        # print("No object found meeting the criteria. Applying Otsu to the entire image.")
        _, output_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    return output_image

def draw_bounding_boxes(color_image, contours):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

def predicate_Q(image, seed_value, y, x, tolerance):
    return np.abs(int(image[y, x]) - int(seed_value)) <= tolerance

def region_grow(image, seeds, tolerance=30):
    h, w = image.shape
    segmented = np.zeros((h, w), dtype=np.int32)
    label_counter = 1
    structure = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]
    
    for seed_y, seed_x in seeds:
        if segmented[seed_y, seed_x] == 0:
            region_queue = deque([(seed_y, seed_x)])
            seed_value = image[seed_y, seed_x]
            segmented[seed_y, seed_x] = label_counter

            while region_queue:
                y, x = region_queue.popleft()
                for dy, dx in structure:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if segmented[ny, nx] == 0 and predicate_Q(image, seed_value, ny, nx, tolerance):
                            segmented[ny, nx] = label_counter
                            region_queue.append((ny, nx))
            label_counter += 1
    return segmented

def is_within_radius(point, center, radius):
    y, x = point
    center_y, center_x = center
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return distance <= radius

def is_within_radius_range(point, center, range_start, range_end):
    distance = np.sqrt((point[1] - center[1]) ** 2 + (point[0] - center[0]) ** 2)
    return range_start <= distance <= range_end

def visualize_centroids(image, centroids, color=(0, 0, 255)):
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for centroid in centroids:
        y, x = centroid
        cv2.circle(color_image, (x, y), 5, color, -1)
    return color_image


def process_image(image_path):
    # image_path = "valid/f--15-_jpg.rf.f9cc0c09bcf07409978fc11d674e94b0.jpg"
    image, thresh_image = load_and_threshold(image_path)
    contours = find_contours(thresh_image)
    areas, bounding_boxes = calculate_areas_and_bounding_boxes(contours)
    selected_object = select_largest_object(areas, bounding_boxes)

    output_image = create_mask_and_apply(image, selected_object)
    contours = find_contours(output_image)
    areas, bounding_boxes = calculate_areas_and_bounding_boxes(contours)
    selected_object = select_largest_object(areas, bounding_boxes)

    color_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)

    contours, hierarchy = cv2.findContours(output_image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    draw_bounding_boxes(color_image, contours)

    if len(image.shape) == 2:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_value = 50
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    centroids = []
    center = (center_y, center_x)

    if bright_image:
        kernel = np.ones((50, 50), np.uint8)
        closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(closed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        areas = [cv2.contourArea(contour) for contour in contours]
        filtered_image = np.zeros_like(closed_image)

        
        seed_points = []

        for i, (contour, area) in enumerate(zip(contours, areas)):
            if 5 < area < 500:
                # print(f"Contour {i}: Area = {area:.2f}")
        
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(filtered_image, (x, y), (x + w, y + h), 255, 2)
                
        
                for point in contour:
                    seed_points.append((point[0][1], point[0][0]))

        if seed_points:
            segmented_region = region_grow(closed_image, seed_points, tolerance=30)
            segmented_region_display = (segmented_region * 255).astype(np.uint8)
            final_image = segmented_region_display

    else:
        if largest_exist:
            areas = [cv2.contourArea(contour) for contour in contours]
            for i, (contour, area) in enumerate(zip(contours, areas)):
                if area > 1:
                    # print(f"Contour {i}: Area = {area:.2f}")
                    for point in contour:
                            x, y = point[0]
                            if binary_image[y, x] == 255 and is_within_radius((y, x), center, radius) * 0.91:
                                centroids.append((y, x))

            outside_points = [
                (y, x) for y in range(image.shape[0]) for x in range(image.shape[1])
                if binary_image[y, x] == 255 and is_within_radius_range((y, x), center, radius * 0.91, radius)
            ]

            if outside_points:
                segmented_outside_image = region_grow(binary_image, outside_points)
                segmented_outside_image_display = np.zeros_like(segmented_outside_image)
                segmented_outside_image_display[segmented_outside_image > 0] = 255

            else:
                segmented_outside_image_display = np.zeros_like(binary_image)
        else:
            segmented_outside_image_display = np.zeros_like(binary_image)
            areas = [cv2.contourArea(contour) for contour in contours]
            for i, (contour, area) in enumerate(zip(contours, areas)):
                if area > 1500:
                    # print(f"Contour {i}: Area = {area:.2f}")
                    for point in contour:
                        x, y = point[0]
                        if binary_image[y, x] == 255:
                                centroids.append((y, x))

    if not bright_image:
        contours, _ = cv2.findContours(segmented_outside_image_display.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area_threshold = 100
        region_growing_seeds = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 <= area <= 1000:
            
                for point in contour:
                    x, y = point[0] 
                    region_growing_seeds.append((y, x)) 

        if region_growing_seeds:
            segmented_region = region_grow(segmented_outside_image_display, region_growing_seeds, tolerance=30)
            segmented_region_display = (segmented_region * 255).astype(np.uint8)
        else:
            segmented_region_display = np.zeros_like(segmented_outside_image_display)




        segmented_image = region_grow(binary_image, centroids)
        segmented_image_display = (segmented_image * 255).astype(np.uint8)

        combined_image = np.zeros_like(segmented_image_display)
        combined_image[segmented_image > 0] = 255

        combined_image[segmented_outside_image_display > 0] = 0

        combined_image[segmented_region_display > 0] = 255



    # plt.subplot(3, 4, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('Original Image')

    # plt.subplot(3, 4, 2)
    # plt.imshow(output_image, cmap='gray')
    # plt.title('Otsu Image')

    # plt.subplot(3, 4, 3)
    # plt.imshow(color_image, cmap='gray')
    # plt.title('Contours')

    # plt.subplot(3, 4, 4)
    # plt.imshow(binary_image, cmap='gray')
    # plt.title('Binary Image')

    # plt.subplot(3, 4, 5)
    # plt.imshow(segmented_region_display, cmap='gray')
    # plt.title('Final result')

    if not bright_image:
        # plt.subplot(3, 4, 5)
        # plt.imshow(segmented_image_display, cmap='gray')
        # plt.title('Result with RG')

        # centroid_image = visualize_centroids(image, centroids, color=(0, 0, 255))
        # plt.subplot(3, 4, 7)
        # plt.imshow(cv2.cvtColor(centroid_image, cv2.COLOR_BGR2RGB))
        # plt.title('Centroids Visualization')

        # plt.subplot(3, 4, 9)
        # plt.imshow(combined_image, cmap='gray')
        # plt.title('Combined Result')


        final_image = combined_image
        contours = find_contours(combined_image)


        if largest_exist:
            # plt.subplot(3, 4, 6)
            # plt.imshow(segmented_outside_image_display, cmap='gray')
            # plt.title('RG - 91% to 100% R')

            centroid_image2 = visualize_centroids(image, outside_points, color=(0, 255, 0))
            # plt.subplot(3, 4, 8)
            # plt.imshow(cv2.cvtColor(centroid_image2, cv2.COLOR_BGR2RGB))
            # plt.title('Outside Points Visualization')
        else:
            kernel = np.ones((3, 3), np.uint8)
            dilated_img = cv2.dilate(combined_image, kernel, iterations=2)

            eroded_img = cv2.erode(dilated_img, kernel, iterations=2)

            contours, _ = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            mask = np.zeros_like(image)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                aspect_ratio = float(w) / h
                if 0.35 <= aspect_ratio <= 1.64:
                    cv2.drawContours(mask, [contour], -1, 255, -1)

            segmented_blocks = cv2.bitwise_and(combined_image, combined_image, mask=mask)
            final_image = segmented_blocks

            # plt.subplot(3, 4, 10)
            # plt.imshow(segmented_blocks, cmap='gray')
            # plt.title('Final Result')

    # plt.show()



    return image, final_image



















def filter_contours_by_area(contours, min_area):
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    return filtered_contours

def fuse_contours(contours, distance_threshold):
    if not contours:
        return []

    all_boxes = [cv2.boundingRect(c) for c in contours]
    fused_boxes = []

    for i in range(len(all_boxes)):
        bx1, by1, bw, bh = all_boxes[i]

        is_fused = False
        for j in range(len(fused_boxes)):
            fb_x1, fb_y1, fb_w, fb_h = fused_boxes[j]
            fb_x2 = fb_x1 + fb_w
            fb_y2 = fb_y1 + fb_h

            if (bx1 <= fb_x2 + distance_threshold and
                bx1 + bw >= fb_x1 - distance_threshold and
                by1 <= fb_y2 + distance_threshold and
                by1 + bh >= fb_y1 - distance_threshold):
                new_x1 = min(fb_x1, bx1)
                new_y1 = min(fb_y1, by1)
                new_x2 = max(fb_x1 + fb_w, bx1 + bw)
                new_y2 = max(fb_y1 + fb_h, by1 + bh)
                fused_boxes[j] = (new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1)
                is_fused = True
                break
        
        if not is_fused:
            fused_boxes.append((bx1, by1, bw, bh))

    return fused_boxes

def draw_bounding_boxes_on_mask(mask_image, min_area=0, distance_threshold=20):
    
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = filter_contours_by_area(contours, min_area)

    fused_boxes = fuse_contours(contours, distance_threshold)

    output_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

    for box in fused_boxes:
        x, y, w, h = box
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    num_contours = len(fused_boxes)
    return output_image, num_contours

def draw_bounding_boxes2(image, annotations):
    output_image = image.copy()
    for index, row in annotations.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Draw in green

    return output_image


def process_and_save_images(input_folder="test_images"):

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    actual = {
        "c--2-_jpg.rf.0326ed404c601fce1ed53b5acba116ed.jpg" : 1,
        "c--18-_jpg.rf.2b2f29301e8141926c07ffb4c85c161b.jpg" : 2,
        "c--70-_jpg.rf.050e808aa9f90f3d0097f57e61887b5f.jpg" : 4,
        "c--45-_jpg.rf.a6a1d41b212d4b9ea7290a4474ed4473.jpg" : 0,
        "c--49-_jpg.rf.c945f2e5ebfb7b16328e4bd970003cb0.jpg" : 3,
        "c--29-_jpg.rf.c745f8305bbad434ef2f033ced947a96.jpg" : 3,
        "c--34-_jpg.rf.e852f0fb16bb5a7204a0cf1670a3da31.jpg" : 2,
        "c--50-_jpg.rf.b3f95f199205f73aaa53e00bfeb2a21d.jpg" : 1,
        "c--33-_jpg.rf.060df43ce21c08b048fc7006cddde82c.jpg" : 0,
        "c--27-_jpg.rf.0caad65ba0c0030d955dcf4d99ad7496.jpg" : 1,
        "c--190-_jpg.rf.00a9250d57514f28a2b092ab8cb4d580.jpg" : 2,
        "c--145-_jpg.rf.d425c871d9e607c5d9f40e841669de0e.jpg" : 2,
        "c--167-_jpg.rf.81951d88c951df31fbc4c991746c25d2.jpg" : 0,
        "c--54-_jpg.rf.dcda4d5fd6da39d739cd613b414c731e.jpg" : 2,
        "c--103-_jpg.rf.2b6feafb66289f2dd300735b293066f7.jpg" : 3,
        "c--91-_jpg.rf.fd4615a24ba1bf4c59222cf13db5c464.jpg" : 1,
        "c--92-_jpg.rf.1bc5c02d6bd4041847097d4f58804260.jpg" : 1,
        "c--108-_jpg.rf.f7659d7bba7f0f55ea596447f18653e0.jpg" : 3,
        "c--169-_jpg.rf.fa7ced6081099b77fa9acec7af792df9.jpg" : 1,
        "c--196-_jpg.rf.006dfcbc6ee82eac637b254b933ff70c.jpg" : 2,
        "c--77-_jpg.rf.d0757ba2fd080cc2bf7eabe8c1ede585.jpg" : 2,
        "c--38-_jpg.rf.6f6d22e248524cc394c14c605ee83238.jpg" : 2,
        "c--35-_jpg.rf.11b7b9f40d845e49e3d656a8c5dc34c5.jpg" : 2,
        "c--47-_jpg.rf.641b989be2eb6c20fcbf9f591cac1f11.jpg" : 1,
        "c--183-_jpg.rf.30414448e73fa023ce6bdf5cc8e536de.jpg" : 3,
        "c--173-_jpg.rf.a61c25de76db954defe8bae82b0837c2.jpg" : 3,
        "c--193-_jpg.rf.9ddaa7a092c26f4866f5ef65e86dce42.jpg" : 4,
        "c--69-_jpg.rf.3957a3460762ebdef8b08e044289b229.jpg" : 4,
        "c--115-_jpg.rf.dac6d7e010e02462379787d7bcbc3a2a.jpg" : 1,
        "c--36-_jpg.rf.095669007899fc6932329c83b4d21c68.jpg" : 4,
        "c--42-_jpg.rf.6ac4e7861933cafe922f3abcf1384eed.jpg" : 3,
        "c--96-_jpg.rf.8d2da905c6aef356ed00ffaee5d88216.jpg" : 3,
        "c--94-_jpg.rf.c627b1f3c4503b49d23df4995bdf2797.jpg" : 2,
        "d--6-_jpg.rf.3c64d69dbe8bbddcfb7691cacc2bccbb.jpg" : 1,
        "d--17-_jpg.rf.22ba2f8b8b408084198875024de87813.jpg" : 3,
        "d--33-_jpg.rf.7b594b748ca1a20c06e43b85bc31bd75.jpg" : 2,
        "d--10-_jpg.rf.a1f6620b15819b628cb1d2c5d9bf9a02.jpg" : 1,
        "d--9-_jpg.rf.482090db96f3b7aff16ddb757f09f6ac.jpg" : 1,
        "d--29-_jpg.rf.80bdfcbb7549c16b410a986ce7fb8d97.jpg" : 3,
        "d--21-_jpg.rf.6f4debba76d18342e361428f8e6b6992.jpg" : 3,
        "d--26-_jpg.rf.9037c5dc255b58eec1148f4570069e3b.jpg" : 2,
        "d--99-_jpg.rf.3f19712821f13dd35d39664efddb6b42.jpg" : 2,
        "d--28-_jpg.rf.6ac3e5875346b295980f47d51d2ffbdd.jpg" : 1,
        "d--90-_jpg.rf.be797c64ec07c91d4e3c10885d756781.jpg" : 3,
        "d--64-_jpg.rf.d74f57aab2740dd26ac7d2d280a0766e.jpg" : 4,
        "d--5-_jpg.rf.59708317c37f56ce648f9c64d6af4651.jpg" : 1,
        "d--24-_jpg.rf.0c0f05aadd997133f7cf2f011b29aee3.jpg" : 3,
        "d--16-_jpg.rf.f5f93d055779c902fed83e37c3280d63.jpg" : 1,
        "d--25-_jpg.rf.3612d09696145227b13e824062a159b0.jpg" : 3,
        "d--23-_jpg.rf.2908633872ad3b5fe524cc540d4f3f2e.jpg" : 2,
        "d--48-_jpg.rf.1de6406b1b8c7e920d956f517c9711e5.jpg" : 2,
        "d--34-_jpg.rf.68e7b8ad1aec84b5f8e3c2838435eba6.jpg" : 2,
        "d--1-_jpg.rf.b37e794762fcd2f017ba3289c0e8183f.jpg" : 1,
        "d--47-_jpg.rf.c3d2afab2860b6f694e63a9feab62d4a.jpg" : 1,
        "d--92-_jpg.rf.a426e9866844833c6da39297392a0fbe.jpg" : 4,
        "d--93-_jpg.rf.c18fd024483919ec196960833ffa6c8d.jpg" : 2,
        "d--14-_jpg.rf.721fb3ecbe01cf8e0ea25aba8aadf5ff.jpg" : 3,
        "d--4-_jpg.rf.81ae2a86ed4fb978be6a19aea9dd76a2.jpg" : 2,
        "d--83-_jpg.rf.705f949f88ba725cc39cf0a559ddfcf3.jpg" : 4,
        "d--97-_jpg.rf.f5e8706299d9b339b6b457fefd93c41f.jpg" : 2,
        "f--26-_jpg.rf.c6a348bd994c2ca70db4e76cde714011.jpg" : 1,
        "f--6-_jpg.rf.a5524b6b396f2e28823c9ab9117c4bd5.jpg" : 4,
        "f--19-_jpg.rf.241fd9188d7232aeda591ceda687df12.jpg" : 1,
        "f--16-_jpg.rf.d2e1b06b454628210af7a8335d9dc8e7.jpg" : 2,
        "f--14-_jpg.rf.a36ddfc1398b21659648730b274087cb.jpg" : 1,
        "f--2-_jpg.rf.48c1fae4ea61bf253c183576ebdddeeb.jpg" : 1,
        "f--1-_jpg.rf.699189e14e6d004d9b979c91ac0e5f9b.jpg" : 2,
        "f--28-_jpg.rf.de4fc5ebba3bbc0baea64e669225c7c5.jpg" : 1,
        "f--15-_jpg.rf.f9cc0c09bcf07409978fc11d674e94b0.jpg" : 2,
        "f--8-_jpg.rf.6294ad4258a2b4cbfd7fd9099a25e506.jpg" : 2,
        "f--30-_jpg.rf.f67245f2b6da827ce462da8693afa6a2.jpg" : 1,
        "f--20-_jpg.rf.9773106d087d869e233fb4fa34c07f1c.jpg" : 2,
        "f--21-_jpg.rf.51e0a86c8c7a52083825fa472f5a1685.jpg" : 2,
        "f--22-_jpg.rf.5d95230e6444fe95c7c1742eab989b4c.jpg" : 1,
        "f--146-_jpg.rf.a810945763576cbf6e4c634833f5f804.jpg" : 2,
        "f--159-_jpg.rf.1a79a131f5bb801957e56d10672aaea7.jpg" : 2,
        "f--194-_jpg.rf.3e113fb70346cbe419d7faea5d190ec1.jpg" : 2,
        "f--191-_jpg.rf.5c36a1a13dbb86362f4e4e777b97918e.jpg" : 1,
        "f--172-_jpg.rf.e8423a42302a5882786f94372b8ed1b2.jpg" : 1,
        "f--155-_jpg.rf.1552cea12cdf8946546803ab306742b2.jpg" : 1,
        "f--153-_jpg.rf.5555f62fbe454aca03c1c7a73838a8cf.jpg" : 0,
        "f--139-_jpg.rf.4eeda4830b617ce5b5a14f8e5681a9d0.jpg" : 2,
        "f--145-_jpg.rf.6fff0983c171566ff671ff960bd11789.jpg" : 1,
        "f--133-_jpg.rf.306c671349e42b165ffff007eaac651f.jpg" : 3,
        "f--130-_jpg.rf.7852150ca73a83d5dc181c0978743e27.jpg" : 1,
        "f--49-_jpg.rf.e5600c4ca8657e0b4f59e4fbc3c1e504.jpg" : 2,
        "f--50-_jpg.rf.09205744ac34b02ecfda68bf236f88a3.jpg" : 1,
        "f--184-_jpg.rf.efad807819e526ec9ae77888e9097106.jpg" : 1,
        "f--110-_jpg.rf.93fd531fbeaf42c7f75340a53f0f292d.jpg" : 2,
        "f--114-_jpg.rf.8e4bec3f92161aca0431fd207dc6fe55.jpg" : 3,
        "f--111-_jpg.rf.f5c6f5877b1112d6009b3ec8dd17d663.jpg" : 2,
        "f--101-_jpg.rf.c9b730a776f016df07324a624fd11533.jpg" : 1,
        "f--91-_jpg.rf.ca99e9200aa3871b41937fd7399bf9e9.jpg" : 3,
        "f--78-_jpg.rf.4a55a812b4d5834c975e0af240fb178a.jpg" : 2,
        "f--45-_jpg.rf.90784e93d46b594cb241ebe5230b390b.jpg" : 1,
        "f--106-_jpg.rf.a7703a859dcddb2f4850fc1f721da0f1.jpg" : 2,
        "f--118-_jpg.rf.e8f9f0608c9cc64125af6a356a284adb.jpg" : 1,
        "f--123-_jpg.rf.0d993b178e0f00f7c3b964078e1ea71c.jpg" : 0,
        "f--129-_jpg.rf.def71fadfc5e0fa09d39531b50408de2.jpg" : 1,
        "f--62-_jpg.rf.d8d97a99d6bb14771779b315bd43d9ad.jpg" : 2,
        "f--71-_jpg.rf.c3b9559de6954fc240457adeab548221.jpg" : 1,
        "f--102-_jpg.rf.3a1621a4cc81d527e810ad781f0f9e4b.jpg" : 0,
        "f--70-_jpg.rf.2f832786d273c15379a9e6bcdb3a0b5f.jpg" : 1,
        "f--66-_jpg.rf.2a381950e452cf44fd2a279ed66bdbff.jpg" : 1,
        "f--112-_jpg.rf.a366549fb0813902426a1091bc6dbe1d.jpg" : 0,
        "f--173-_jpg.rf.a53617ffec15f00b4a1e9830e4a0a16d.jpg" : 2
    }

    predictions = []
    y_true = []
    y_pred = []

    passed = 0
    failed = 0

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        original_image, final_image = process_image(image_path)

        min_area = 20
        distance_threshold = 20 if largest_exist else 0

        result_image, num_contours = draw_bounding_boxes_on_mask(final_image, min_area, distance_threshold)

        predictions.append((image_file, num_contours))

        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.title(image_file)
        # plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')

        # plt.subplot(1, 2, 2)
        # plt.title(f'Contours: {num_contours}')
        # plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))  # Display final image
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()
        
        
        y_true.append(actual.get(image_file, 0))
        y_pred.append(num_contours)

        if image_file in actual:
            if actual[image_file] == num_contours:
                passed += 1
            else:
                failed += 1

    report = classification_report(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\nTotal Passed: {passed}")
    print(f"Total Failed: {failed}")

    print(f"Classification Report:\n{report}")
    print(f"Accuracy: {accuracy:.2f}")
    
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(set(y_true)), 
                yticklabels=sorted(set(y_true)))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
    

process_and_save_images()