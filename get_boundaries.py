import cv2
import numpy as np

GENERAL_MGV_THRESHOLD = 187 #Threshold of Mean Gray Value that define that the image is ambiguous so it is hard to find boundaries (higher value - more white/empty parts are allowed in the image)
MGV_VALUE_PROPORTION = 0.83 #Value that needs to be used for defining adaptive mgv_threshold, you can experiment and customize it for a specific type of images

def get_boundaries (img_source, chunk_len=False):
    def increase_contrast(img):
        array_alpha = np.array([1.25])
        array_beta = np.array([-100.0])
        cv2.add(img, array_beta, img)                    
        cv2.multiply(img, array_alpha, img)
        return img
    def increase_brightness(img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    def crop_image (img, chunk_len):
        height, width = img.shape[:2]
        num_blocks = 500
        if not chunk_len:
            chunk_len = int(min ([height, width])/(num_blocks**0.5)) #Find adaptive chunk_len based on min side len
        
        chunks = {}
        for x in range(0, width, chunk_len):
            for y in range(0, height, chunk_len): 
                y1 = y
                y2 = y + chunk_len
                x1 = x
                x2 = x + chunk_len
                chunk = img[y1:y2, x1:x2]
                chunks [str (x)+' '+str (y)] = chunk
        return chunks, chunk_len
    def get_mgv (img): #Mean Gray Value
        mgv = cv2.mean(img)[0]
        return mgv
    def restore_mask (mask_orig, mask_chunks, chunk_len):
        for coords in list(mask_chunks.keys()):
            xy = [int (coord) for coord in coords.split (' ')]
            x1 = xy [0]
            y1 = xy [1]
            y2 = y1 + chunk_len
            x2 = x1 + chunk_len
            mask_orig[y1:y2, x1:x2] = mask_chunks [coords]
        return mask_orig
    def fill_spaces (mask, mask_chunks, chunk_len):
        def transform_needed (mask, x1, y1, x2, y2, chunk_len, height, width, white_color): #Check if any of two neighboring blocks are white or all black - needs transformation
            #New y1, y2, x1, x2
            if y1-chunk_len > 0:
                upper_coords = [y1-chunk_len, y1, x1, x2]
            else:
                upper_coords = [0, y1, x1, x2]
            if y2+chunk_len < height:
                lower_coords = [y2, y2+chunk_len, x1, x2]
            else: 
                lower_coords = [y2, height, x1, x2]
            if x1-chunk_len > 0:
                left_coords = [y1, y2, x1-chunk_len, x1]
            else:
                left_coords = [y1, y2, 0, x1]
            if x2+chunk_len < width:
                right_coords = [y1, y2, x2, x2+chunk_len]
            else:  
                right_coords = [y1, y2, x2, width]
            
            #Check what color to search
            if white_color:
                color_to_search = 0
            else:
                color_to_search = 255

            block_counter = 0
            try:
                if mask[upper_coords [0]:upper_coords [1], upper_coords [2]:upper_coords [3]][0][0] == color_to_search:
                    block_counter += 1
            except IndexError:
                pass
            try:
                if mask[lower_coords [0]:lower_coords [1], lower_coords [2]:lower_coords [3]][0][0] == color_to_search:
                    block_counter += 1
            except IndexError:
                pass
            try:
                if mask[left_coords [0]:left_coords [1], left_coords [2]:left_coords [3]][0][0] == color_to_search:
                    block_counter += 1
            except IndexError:
                pass
            try:
                if mask[right_coords [0]:right_coords [1], right_coords [2]:right_coords [3]][0][0] == color_to_search:
                    block_counter += 1
            except IndexError:
                pass
            
            #If it is a white block and all around are black - turn it black
            if white_color:
                if block_counter == 4:
                    return True
                else:
                    return False
            #If it is a black block and at least 2 around are white - turn it white
            else:
                if block_counter >= 2:
                    return True
                else:
                    return False

        height, width = mask.shape[:2]
        for coords in list(mask_chunks.keys()):
            xy = [int (coord) for coord in coords.split (' ')]
            x1 = xy [0]
            y1 = xy [1]
            y2 = y1 + chunk_len
            x2 = x1 + chunk_len

            if mask_chunks [coords][0][0] == 0: #Define block color
                white_color = False
            else:
                white_color = True

            #If it is a not needed block - decide if need to turn white, or if needed - to turn black
            if transform_needed (mask, x1, y1, x2, y2, chunk_len, height, width, white_color):
                chunk_height, chunk_width = mask_chunks [coords].shape[:2]
                img = np.zeros([chunk_height, chunk_width],dtype=np.uint8)

                if white_color:
                    img.fill(0)
                else: 
                    img.fill(255)
                mask[y1:y2, x1:x2] = img
        return mask
    
    if type(img_source) != str: #Load image
        nparr = np.frombuffer(img_source, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        img_cv = cv2.imread(img_source)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    mask_image = increase_contrast(img_cv) #Set contrast
    mask_image = increase_brightness(mask_image, value=10) #Set brightness
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY) #Turn into gray scale
    mask_image = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] #Apply threshold
    mask_image = cv2.bitwise_not(mask_image) #Invert

    #Define mgv threshold
    general_mgv = int(get_mgv (img_cv))
    if general_mgv >= GENERAL_MGV_THRESHOLD: #Means all data needs to be returned
        height, width = mask_image.shape[:2]
        mask_image = np.zeros([height, width],dtype=np.uint8)
        mask_image.fill(255)
        return mask_image, img_cv
    else:
        mgv_threshold = int (general_mgv/MGV_VALUE_PROPORTION)

    cropped_imgs, chunk_len = crop_image (mask_image, chunk_len) #Get smaller chunks of image
    for coords in list(cropped_imgs.keys()):
        mgv = get_mgv (cropped_imgs [coords]) #Get mean gray value
        height, width = cropped_imgs [coords].shape[:2]

        if mgv >= mgv_threshold: #Not needed
            img = np.zeros([height, width],dtype=np.uint8)
            img.fill(0)
            cropped_imgs [coords] = img
        else: #Needed
            img = np.zeros([height, width],dtype=np.uint8)
            img.fill(255)
            cropped_imgs [coords] = img
    mask_image = restore_mask (mask_image, cropped_imgs, chunk_len) #Restore mask based on chunks
    mask_image = fill_spaces (mask_image, cropped_imgs, chunk_len) #Fill empty spaces between mask parts

    fg = cv2.bitwise_or(img_cv, img_cv, mask=mask_image)
    mask = cv2.bitwise_not(mask_image)
    background = np.full(img_cv.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(background, background, mask=mask)

    final_image = cv2.bitwise_or(fg, bk)
    return mask_image, final_image