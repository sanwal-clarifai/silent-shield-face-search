import os
import pandas as pd
import streamlit as st
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format
import base64
import matplotlib.pyplot as plt
import io
from google.protobuf.json_format import MessageToDict
import requests
from PIL import Image
from io import BytesIO

auth = ClarifaiAuthHelper.from_streamlit(st)
print(auth._pat)
stub = auth.get_stub()
userDataObject = auth.get_user_app_id_proto()
userDataClarifaiMain= resources_pb2.UserAppIDSet(user_id='clarifai', app_id='main')

def crop_workflow_predict(file_bytes,
                     WORKFLOW_ID="Face-detect-crop"):
    # Crop the face with this workflow to feed to visual search
   
    post_workflow_results_response = stub.PostWorkflowResults(
        service_pb2.PostWorkflowResultsRequest(
            user_app_id=userDataObject,
            workflow_id=WORKFLOW_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=file_bytes
                        )
                    )
                )
            ]
        ),
        metadata=(('authorization', 'Key ' + auth._pat),)
    )
    return post_workflow_results_response

def make_visual_search_req(file_bytes):
    
    post_annotations_searches_response = stub.PostAnnotationsSearches(
        service_pb2.PostAnnotationsSearchesRequest(
            user_app_id=userDataObject,
            searches=[
                resources_pb2.Search(
                    query=resources_pb2.Query(
                        ranks=[
                            resources_pb2.Rank(
                                annotation=resources_pb2.Annotation(
                                    data=resources_pb2.Data(
                                        image=resources_pb2.Image(
                                            base64=file_bytes
                                        )
                                    )
                                )
                            )
                        ]
                    )
                )
            ]
        ),
        metadata=(('authorization', 'Key ' + auth._pat),)
    )
    return post_annotations_searches_response

def predict_demographics(file_bytes):
    DEM_USER_ID = 'clarifai'
    # Your PAT (Personal Access Token) can be found in the portal under Authentification
    PAT = '28f330bd4e254e5fafe0dae0a9407c25'
    DEM_APP_ID = 'main'
    # Change these to make your own predictions
    DEM_WORKFLOW_ID = 'Demographics'
    dem_userDataObject = resources_pb2.UserAppIDSet(user_id=DEM_USER_ID,
                                                     app_id=DEM_APP_ID) # The userDataObject is required when using a PAT

    post_workflow_results_response = stub.PostWorkflowResults(
        service_pb2.PostWorkflowResultsRequest(
            user_app_id=dem_userDataObject,  
            workflow_id=DEM_WORKFLOW_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=file_bytes
                        )
                    )
                )
            ]
        ),
        metadata=(('authorization', 'Key ' + auth._pat),)
    )
    return post_workflow_results_response

def face_sentiment(file_bytes):
    # Your PAT (Personal Access Token) can be found in the portal under Authentification
    PAT = '28f330bd4e254e5fafe0dae0a9407c25'
    # Specify the correct user_id/app_id pairings
    # Since you're making inferences outside your app's scope
    fs_USER_ID = 'clarifai'
    fs_APP_ID = 'main'
    # Change these to whatever model and image URL you want to use
    MODEL_ID = 'face-sentiment-recognition'

    fs_userDataObject = resources_pb2.UserAppIDSet(user_id=fs_USER_ID, app_id=fs_APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=fs_userDataObject,  
            model_id=MODEL_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=file_bytes
                        )
                    )
                )
            ]
        ),
        metadata=(('authorization', 'Key ' + auth._pat),)
    )
    return post_model_outputs_response



if __name__ == '__main__':

    st.set_page_config(layout="wide")
    st.title("Finding Person of interest with Clarifai")

    with st.sidebar:
        st.sidebar.markdown("**Filter Search matches below this value**", unsafe_allow_html=True)
        slider_value = st.sidebar.slider("Set Threshold for face matches", 0.0, 1.0, 0.5)
        age_slider_value = st.sidebar.slider("**Age**", 0.0, 1.0, 0.5)
        gender_slider_value = st.sidebar.slider("**Gender**", 0.0, 1.0, 0.5)
        ethnic_slider_value = st.sidebar.slider("**Ethnic Appearance**", 0.0, 1.0, 0.5)
        sentiment_slider_value = st.sidebar.slider("**Face Sentiment**", 0.0, 1.0, 0.5)    
    
    
    uploaded_file = st.file_uploader("Choose an image")#, on_change=reset_session_state)
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
       
        # Display the image
        image = Image.open(io.BytesIO(file_bytes))
        st.image(image, caption='Uploaded Image')

        # Print image dimensions
        image_width, image_height = image.size
        st.write(f"**Image dimensions**: {image_width} x {image_height}")
        

        # Now predict with the workflow where we detect + crop faces
        detect_faces_crop = crop_workflow_predict(file_bytes)
        if detect_faces_crop.status.code != status_code_pb2.SUCCESS:
            print(detect_faces_crop.status)
            print("Post workflow results failed, status: " + detect_faces_crop.status.description)

        # st.write(detect_faces_crop.results[0].outputs[-1].data.regions)
        detected_faces = detect_faces_crop.results[0].outputs[1].data.regions

        st.write(f'**Detected faces**: {len(detected_faces)}')
        for face in detected_faces:            
            # SAVE THE CROPPED FACE AS A BASE64 STRING
            st.write('-'*100)
            base_64_img = face.data.image.base64
            st.image(base_64_img, caption='Cropped Face')
            # Send the cropped face to visual search
            with st.spinner("Searching for this face"):
                vis_search_req = make_visual_search_req(base_64_img)
                urls = []
                scores = []
                names = []
                df= pd.DataFrame(columns=['url','input_type','score', 'metadata' ])
                for i in range(len(vis_search_req.hits)):
                    if 'image' in MessageToDict(vis_search_req.hits[i].input.data):
                        img_url = vis_search_req.hits[i].input.data.image.url
                        input_type = 'image'
                        frame_time_string = '-'
                    else:
                        img_url = vis_search_req.hits[i].input.data.video.url
                        
                        video_frames = vis_search_req.hits[i].annotation.data.frames
                        delimiter = ":"
                        frame_time_list = []

                        for frame in video_frames:
                            # Convert frame time to seconds
                            frame_time = (frame.frame_info.time-500)/1000

                            # Convert seconds to minutes and seconds
                            minutes, seconds = divmod(frame_time, 60)

                            # Format the time as mm:ss
                            frame_time_formatted = f"{int(minutes):02d}:{int(seconds):02d}"

                            frame_time_list.append(frame_time_formatted)

                        # Join the formatted frame times with the delimiter
                        frame_time_string = delimiter.join(frame_time_list)
                        input_type = 'video'

                    matches = vis_search_req.hits[i]
                    score = vis_search_req.hits[i].score

                    metadata = vis_search_req.hits[i].input.data.metadata

                    if 'img_name' in metadata.keys():
                        name = metadata['img_name']
                    elif 'filename' in metadata.keys():
                        name = metadata['filename']
                    else:
                        name = metadata

                    df.loc[len(df)] = {'url': img_url, 'input_type': input_type, 'frame time': frame_time_string,
                                       'score': score, 'metadata': name}
                    urls.append(img_url)
                    scores.append(score)
                filtered_df = df[df['score'] > slider_value]
                st.subheader("**Clarifai search results for this face**")
                st.dataframe(filtered_df)
                # Determine the number of items to display, up to a maximum of 10
                num_items_to_display = min(len(filtered_df), 10)

                # Display the first row of up to 5 items
                first_row_items = min(num_items_to_display, 5)
                first_row_cols = st.columns(5)
                for i in range(first_row_items):
                    row = filtered_df.iloc[i]
                    with first_row_cols[i]:
                        url = row['url']
                        input_type = row['input_type']
                        score = round(row['score'], 3)
                        metadata = row['metadata']

                        # Create a Markdown string with each item on a new line
                        markdown_link = f"Score: {score} <br><br> {metadata} <br> [URL]({url})"
                        st.markdown(markdown_link, unsafe_allow_html=True)

                        # Display the content based on its type
                        if input_type == 'image':
                            st.image(url, use_column_width=True)
                        elif input_type == 'video':
                            st.video(url)

                # Display the second row if there are more than 5 items
                if num_items_to_display > 5:
                    second_row_items = num_items_to_display - 5
                    second_row_cols = st.columns(5)
                    for i in range(second_row_items):
                        row = filtered_df.iloc[i + 5]
                        with second_row_cols[i]:
                            url = row['url']
                            input_type = row['input_type']
                            score = round(row['score'], 3)
                            metadata = row['metadata']

                            # Create a Markdown string with each item on a new line
                            markdown_link = f"Score: {score} <br><b4> {metadata} <br> [URL]({url})"
                            st.markdown(markdown_link, unsafe_allow_html=True)

                            # Display the content based on its type
                            if input_type == 'image':
                                st.image(url, use_column_width=True)
                            elif input_type == 'video':
                                st.video(url)


            # Send the cropped face to demographics
            with st.spinner("Sending to Demographics analysis"):
                demographics_req = predict_demographics(base_64_img)   
                # We'll get one WorkflowResult for each input we used above. Because of one input, we have here one WorkflowResult
                dem_results = demographics_req.results[0]
                ethnicity = dem_results.outputs[2].data.regions
                gender = dem_results.outputs[3].data.regions
                age = dem_results.outputs[4].data.regions

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.write("**Ethnicity:Score**")
                    ethnicity_df = pd.DataFrame([(concept.name, concept.value) for concept in ethnicity[0].data.concepts], columns=['Ethnicity', 'Score'])
                    ethnicity_df  = ethnicity_df[ethnicity_df['Score']>ethnic_slider_value]
                    st.dataframe(ethnicity_df.style.set_properties(**{'overflow-y': 'scroll', 'height': '300px', 'width': '150%'}))

                with col2:
                    st.write("**Gender:**")
                    gender_df = pd.DataFrame([(concept.name, concept.value) for concept in gender[0].data.concepts], columns=['Gender', 'Score'])
                    gender_df = gender_df[gender_df['Score']>gender_slider_value]
                    st.dataframe(gender_df.style.set_properties(**{'overflow-y': 'scroll', 'height': '300px', 'width': '150%'}))

                with col3:
                    st.write("**Age:**")
                    age_df = pd.DataFrame([(concept.name, concept.value) for concept in age[0].data.concepts], columns=['Age', 'Score'])
                    age_df = age_df[age_df['Score']>age_slider_value]
                    st.dataframe(age_df.style.set_properties(**{'overflow-y': 'scroll', 'height': '300px', 'width': '150%'}))
                
                with col4:
                    sentiment_prediction = face_sentiment(base_64_img)
                    sentiment_results = sentiment_prediction.outputs[0]
                    # sentiment = sentiment_results.outputs[0].data.regions
                    sentiment_results_df = pd.DataFrame([(concept.name, concept.value) for concept in sentiment_results.data.concepts], columns=['Sentiment', 'Score'])
                    sentiment_results_df = sentiment_results_df[sentiment_results_df['Score']>sentiment_slider_value]
                    st.write("**Sentiment:**")
                    st.dataframe(sentiment_results_df.style.set_properties(**{'overflow-y': 'scroll', 'height': '300px', 'width': '150%'}))
                
               

                    
            
           