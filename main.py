import cv2
import json
import textwrap
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from utils import docs
st.set_page_config(layout="wide")
from streamlit_option_menu import option_menu   
from streamlit_extras.colored_header import colored_header
from image_annotation import run_cls, dataframe_annotation
from dicom_viewer_and_annon import anonymize_dicom_file, dicom_viewer
from image_enhancement import clahe_image_enhance, increase_brightness,gamma,super_resolution, noise_reduction,threshold_intensity,median
from src.full_model.generate_reports_for_images import main_model
# ignore warnings
warnings.filterwarnings("ignore")

def features():
    """
    Function to handle the different features of the application.
    """
    #  headings and tabs creation
    DicomAnonymizationTab, ImgAnnotationTab, ImgEnhancementTab, PredictionTab,  = st.tabs(["Dicom Image Explorer and Anonymizer", 
    "Image Annotation and labelling", "Image Enhancement", "Prediction and Report Generation", ])
    
    # DicomAnnonymizationTab: Here is the entry point for dicom image explorer
    with DicomAnonymizationTab:
        DcmAnonTab, DcmViewTab, = st.tabs(["Dicom Image Anonymization", "Dicom Viewer",])
        
        # DcmAnonTab: Here is the entry point for dicom image annonymization
        with DcmAnonTab:
            
            # upload a dicom file
            dicom_file = st.file_uploader("Upload a DICOM file", key="dicom_annon")
            
            # checks if file has been uploaded
            if dicom_file is not None: 
                
                # uses anonymize_dicom_file to anonymize DICOM file
                anonymized_dicom_dataset = anonymize_dicom_file(dicom_file)
                desired_file_name = st.text_input("Enter the desired file name for the anonymized DICOM file, press enter to save: ")
                
                # Save the anonymized DICOM file
                if st.button("Save anonymized dicom file"):
                    if desired_file_name is not None:
                        if not desired_file_name.endswith('.dcm'):
                            # Add the .dcm extension if not provided
                            desired_file_name += '.dcm'
                            anonymized_dicom_dataset.save_as(f'data/{desired_file_name}')
                            st.success('Anonymized Dicom file Saved!')
        
        # DcmViewTab: Here is the entry point for dicom image viewer
        with DcmViewTab:
            # upload a dicom file
            dicom_file = st.file_uploader("Upload a DICOM file", key="dicom_viewer")
            # checks if file has been uploaded
            if dicom_file is not None:
                # uses dicom_viewer function to view uploaded file
                dicom_viewer(dicom_file)
    
    # ImgenhancementTab: Here is the entry point for SOTA Image enhancement
    with ImgEnhancementTab:
        # define path
        path = './data/'
        
        # upload a file
        image = st.file_uploader('Upload an image for enhancement')

        # checks if file has been uploaded
        if image:
            # asks user for an algorithm
            option = st.selectbox(
                'Select an algorithm for image enhancement',
                ('Contrast limited adaptive histogram equalization','Brightness', 'Histogram Equalization','Gamma correction','Super Resolution','Smoothing','Thresholding Intensity','Median Filter'), index=1)
            
            # create colums and displays original image on the left
            col_1, col_2 = st.columns(2)
            col_1.image(image, caption='Original Image', use_column_width=True)

            # Uses the selected options to transform image
            if option == 'Contrast limited adaptive histogram equalization':
                if col_1.button('Click to perform enhancement'):
                    output_image = clahe_image_enhance(path + image.name, mode='CLHE')
                    col_2.image(output_image, caption='CLAHE Enhanced Image')
                # saving result
                if col_2.button('Save Enhanced Image'):
                    cv2.imwrite(f'{path}{image.name.split(".")[0]}_CLAHE_enhanced.jpeg', clahe_image_enhance(path + image.name, mode='CLHE'))
                    st.success(f"Image with Contrast limited adaptive histogram equalization Enhancement saved!")
            
            # Uses the selected options to transform image
            elif option == 'Histogram Equalization':
                if col_1.button('Click to perform enhancement'):
                    output_image = clahe_image_enhance(path + image.name, mode='HE')
                    col_2.image(output_image, caption='HE Enhanced Image')
                # saving result
                if col_2.button('Save Enhanced Image'):
                    cv2.imwrite(f'{path}{image.name.split(".")[0]}_HE_enhanced.jpeg', clahe_image_enhance(path + image.name, mode='HE'))
                    st.success(f"Image with Contrast limited adaptive histogram equalization Enhancement saved!")

            elif option == 'Brightness':
                value = st.slider('Increase brightness with this slide bar',0,255)
                output_image = increase_brightness(path + image.name,value)
                col_2.image(output_image, caption='Brightness')
                # saving result
                if col_2.button('Save Enhanced Image'):
                    cv2.imwrite(f'{path}{image.name.split(".")[0]}_increase_brightness_enhanced.jpeg', increase_brightness(path + image.name))
                    st.success(f"Image with ssr saved!")

            elif option == 'Gamma correction':
                if col_1.button('Click to perform enhancement'):
                    output_image = gamma(path + image.name)
                    col_2.image(output_image, caption='Gamma Corrected Image')

                if col_2.button('Save Enhanced Image'):
                    cv2.imwrite(f'{path}{image.name.split(".")[0]}_gamma_corrected.jpeg',gamma(path + image.name))
                    st.success(f"Image with Gamma Correction Enhancement saved!")

            elif option == 'Super Resolution':
                if col_1.button('Click to perform enhancement'):
                    output_image = super_resolution(path + image.name, upscale_factor=4)
                    col_2.image(output_image, caption='Super Resolved Image')
                if col_2.button('Save Enhanced Image'):
                    cv2.imwrite(f'{path}{image.name.split(".")[0]}_super_resolved.jpeg',super_resolution(path + image.name))
                    st.success(f"Image with Super Resolution saved!")

            elif option == 'Smoothing':
                if col_1.button('Click to perform enhancement'):
                    output_image = noise_reduction(path + image.name, sigma=1.2)
                    col_2.image(output_image, caption='Noise-Reduced Image')
                if col_2.button('Save Enhanced Image'):
                    cv2.imwrite(f'{path}{image.name.split(".")[0]}_noise-reduced.jpeg',noise_reduction(path + image.name))
                    st.success(f"Image with Smoothing saved!")

            elif option == 'Thresholding Intensity':
                value = st.slider('Increase threshold with this slide bar', 0, 200)
                output_image = threshold_intensity(path + image.name,value)
                col_2.image(output_image, caption='Thresholded Image')
                if col_2.button('Save Enhanced Image'):
                    cv2.imwrite(f'{path}{image.name.split(".")[0]}_threshold_intensity.jpeg',
                                threshold_intensity(path + image.name,value))
                    st.success(f"Image with Median filter saved!")


            elif option == 'Median Filter':
                if col_1.button('Click to perform enhancement'):
                    output_image = median(path + image.name)
                    col_2.image(output_image, caption='Median Enhanced Image')
                # saving result
                if col_2.button('Save Enhanced Image'):
                    cv2.imwrite(f'{path}{image.name.split(".")[0]}_Median_enhanced.jpeg',
                                median(path + image.name))
                    st.success(f"Image with Median Filter Enhancement saved!")

    # ImgAnnotationTab: Here is the entry point for image annotation
    with ImgAnnotationTab:
        # define labels
        custom_labels = ["", "Lesion", "Positive", "Negative", "Tumor","Pneumonia", "Covid", None]
        # gets directory from user
        path = st.text_input('Enter the path to image folder', key="clsTab_path")
        # checks if path has been given
        if path:
            # uses run_cls function for annotation
            select_label, report = run_cls(f"{path}", custom_labels)
            # saves generated datadrame
            dataframe_annotation(f'{path}/*.jpg', custom_labels, select_label, report)
    
    # FinetuningTab: Here is the entry point for model finetuning
    # with FinetuningTab:
    #     st.write('wait a minute')
    
    # PredictionTab: Here is the entry point for report generation
    with PredictionTab:
        # upload an image file
        uploaded_image = st.file_uploader('Upload file for Report Generation!')
        
        # converts and saves BGR (3 channel) images to grayscale
        def coloured_to_gray_scale(input_image: str, path:str):
            """  
            Converts and saves BGR (3 channel) images to grayscale.

            Args:
                input_image (str): Path to the input image file.
                path (str): Path to the directory where the grayscale image will be saved.

            Returns:
                bool: True if the grayscale image was successfully saved, False otherwise.
            """
            img = cv2.imread(input_image)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.imwrite(f'{path}{input_image.split("/")[2].split(".")[0]}_gray_scaled.jpg', gray_image)
       
        # define path 
        path = "./data/"
        col_1, col_2 = st.columns(2)   
        
        # checks if file has been uploaded
        if uploaded_image:
            # displays original image on the left
            col_1.image(uploaded_image)
            coloured_to_gray_scale(f"{path}{uploaded_image.name}", path)
            if col_1.button('Generate Report'):   
                # uses model to generate and display report on the right
                report = main_model(f"{path}{uploaded_image.name.split('.')[0]}_gray_scaled.jpg")
                col_2.write(report)


def main():
    ##st.markdown("<h2 style='text-align: center; color: blue;'>Smart Chest-Xray Analysis and Report Generation!</h2>", unsafe_allow_html=True)
    """
    Main function to run the application.
    """
    
    # sidebar: used option_menu just for asthetics
    with st.sidebar:
        choice = option_menu("Main Menu", ["About", "Try out!"], 
            icons=['house', 'fire'], menu_icon="cast", default_index=0,
        styles={
        "container": {"padding": "0!important", "background-color": "#262730"},
        "icon": {"color": "white", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#3739b5"},
        "nav-link-selected": {"background-color": "#1f8ff6"},})  

    # navigations 
    if choice == "About":
        st.image('utils/CI 1.png', use_column_width=True)
        st.write('Write short documentation here')
        docs() 
 
    elif choice == "Try out!":
        colored_header(
        label="CHEST-INSIGHT: Smart Chest-Xray Analysis and Report Generation! ",
        description="Use the tabs below to tryout our dedicated tools",
        color_name="violet-70",)
        features()


if __name__ == "__main__":
    main()
