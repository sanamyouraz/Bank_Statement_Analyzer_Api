import cv2
import numpy as np
from pdf2image import convert_from_path
import os
from PIL import Image,ImageFont
from paddleocr import PPStructure,draw_structure_result,save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx


def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply morphological operations to clean the image (optional)
    kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

    # Other preprocessing steps can be added here, such as resizing, denoising, etc.

    return cleaned_image

def process_and_save_images(input_file, output_dir):
    images = convert_from_path(input_file)

    for idx, image in enumerate(images):
        processed_image = preprocess_image(np.array(image))

        # Apply pixel manipulation logic
        height, width = processed_image.shape
        for y in range(height):
            for x in range(width):
                if processed_image[y, x] <= 208 and processed_image[y, x] >= 50:
                    processed_image[y, x] = 50

        output_image_path = os.path.join(output_dir, f"page{idx+1}.jpg")
        cv2.imwrite(output_image_path, processed_image)\









if __name__ == "__main__":
    input_file = r"statement_pdf/statements.pdf"
    output_directory = r"C:\Users\sanam\PycharmProjects\Bank_Statement_Analyzer_Api\preprocessed_image"

    process_and_save_images(input_file, output_directory)


table_engine2 = PPStructure(show_log=True, image_orientation=True)
table_engine = PPStructure(show_log=True, image_orientation=True,layout_score_threshold = 0.7,ser_model_dir='ser/ser_LayoutXLM_xfun_zh',structure_version='PP-StructureV2')




save_folder = 'output/'  #output of ocr

# Folder path containing the images
folder_path = 'preprocessed_image/'
# Get a list of image filenames from the folder
image_filenames = os.listdir(folder_path)
# Loop through the list of image filenames and process each image
for filename in image_filenames:
    img_path = os.path.join(folder_path, filename)
    img = cv2.imread(img_path)
    result = table_engine(img)
    save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

    for line in result:
        line.pop('img')
        print(line)
    font_path2 = 'doc/fonts/simfang.ttf'  # PaddleOCR下提供字体包
    image = Image.open(img_path).convert('RGB')
    font_size = 12  # Replace with your desired font size
    # Create the ImageFont object using the font path and font size
    font = ImageFont.truetype(font_path2, font_size)
    im_show = draw_structure_result(image, result, font_path=font_path2)
    im_show = Image.fromarray(im_show)
    # Extract the image name without extension
    image_name = os.path.basename(img_path).split('.')[0]

    # Save the processed image
    im_show.save(os.path.join('page_image_with_orientation', f'result_{image_name}.jpg'))

    h, w, _ = img.shape
    res = sorted_layout_boxes(result, w)
    convert_info_docx(img, res, save_folder, image_name)




