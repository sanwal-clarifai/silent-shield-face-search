import datetime
import streamlit as st
import pandas as pd
from datetime import timedelta

from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai_utils.modules.css import ClarifaiStreamlitCSS
from clarifai_grpc.channel.custom_converters.custom_dict_to_message import dict_to_protobuf
from clarifai_grpc.channel.custom_converters.custom_message_to_dict import protobuf_to_dict

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_pb2, status_code_pb2
from google.protobuf.struct_pb2 import Struct
from google.protobuf.json_format import MessageToDict

import matplotlib.pyplot as plt
import os
from utils.api_utils import *


CLARIFAI_BLUE = '#356dff'
CLARIFAI_RGB_OP_50 = 'rgba(53,109,255, 0.5)'

try:
  st.set_page_config(layout="wide")
except:
  pass

ClarifaiStreamlitCSS.insert_default_css(st)
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = auth.get_stub()
metadata = auth.metadata
authkey = metadata[0]# get the tuple that is either auth key or session token
# authkey = dict([authkey])# convert tuple to dict
base_url = "https://" + auth._base
userDataObject = auth.get_user_app_id_proto()

if 'evaluate_clicked' not in st.session_state:
    st.session_state.evaluate_clicked = False


@st.cache_data
def find_matches(file_bytes,min_confidence_score):
    all_matches = []
    ok_to_continue = True
    current_page = 1
    max_per_page = 1000
    while(ok_to_continue):
        post_annotations_searches_response = stub.PostAnnotationsSearches(
            service_pb2.PostAnnotationsSearchesRequest(
                user_app_id=userDataObject,
                searches = [
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
                ],
                pagination=service_pb2.Pagination(page=current_page, per_page=max_per_page),
            ),
            metadata=metadata
        )

        if post_annotations_searches_response.status.code != status_code_pb2.SUCCESS:
            print(post_annotations_searches_response.status)
            raise Exception("Post searches failed, status: " + post_annotations_searches_response.status.description)


        prediction_response = MessageToDict(post_annotations_searches_response)
        last_in_list = len(prediction_response['hits']) - 1
        for hits in prediction_response['hits']:
            if hits['score'] >= min_confidence_score:
                all_matches.append(hits)
        current_page = current_page + 1
        if (current_page * max_per_page) > 4000:
            ok_to_continue = False
        if prediction_response['hits'][last_in_list]['score'] < min_confidence_score:
            ok_to_continue = False
        if len(prediction_response['hits']) < max_per_page:
            ok_to_continue = False

    return all_matches


def total_processed_inputs():
    get_input_count_response = stub.GetInputCount(
        service_pb2.GetInputCountRequest(
            user_app_id=userDataObject            
        ),
        metadata=metadata
    )
    if get_input_count_response.status.code != status_code_pb2.SUCCESS:
        print(get_input_count_response.status)
        raise Exception("Post searches failed, status: " + get_input_count_response.status.description)
    return get_input_count_response.counts.processed


@st.cache_data
def generate_dataframe(prediction_response, confidence):
    raw_data = []
    app_url = f"https://clarifai.com/{auth.user_id}/{auth.app_id}/inputs/"
    for hit in prediction_response:
        if hit['score'] < confidence:
            next
        input_url = f"{app_url}{hit['input']['id']}"
        if 'image' in hit['input']['data']:
            raw_data.append(['image', hit['input']['data']['image']['url'], hit['score'], None, input_url, None])
            # raw_data.append(['image', hit['input']['data']['image']['url'], hit['score']])
        if 'video' in hit['input']['data']:
            found_time_ms = hit['annotation']['data']['frames'][0]['frameInfo']['time']
            converted_time = str(timedelta(seconds=(found_time_ms/1000)))
            if 'filename' in hit['input']['data']['metadata']:
                file_base = hit['input']['data']['metadata']['filename']
                raw_data.append(['video', hit['input']['data']['video']['url'], hit['score'], converted_time, input_url, file_base])
            else:
                raw_data.append(['video', hit['input']['data']['video']['url'], hit['score'], converted_time, input_url, hit['input']['data']['video']['url']])
    return pd.DataFrame(raw_data, columns = ['Type', 'URL', 'Score', 'TimeIndex', 'Input URL', 'Segment']).drop_duplicates()


@st.cache_data
def detail_responses(df, sort_order):
    # df = df.astype({'TimeIndex':pd.StringDtype()})
    details = df.copy()
    if sort_order == 'Similarity':
        details = df.sort_values(by=['Score'],ascending=False)
    if sort_order == 'Frame Count':
        details = df.sort_values(by=['Segment'],ascending=False)
    details = details.reset_index(drop=True)
    return details


def aggregate_responses_by_timeindex(df, sort_order):
    initial_aggregation = df.groupby(['Type', 'URL','Segment','TimeIndex']).agg({'Score':['count','mean','min','max']})
    if sort_order == 'Similarity':
        initial_aggregation = initial_aggregation.sort_values(by=('Score', 'max'),ascending=False)
    if sort_order == 'Frame Count':
        initial_aggregation = initial_aggregation.sort_values(by=('Score', 'count'),ascending=False)
    initial_aggregation = initial_aggregation.reset_index()
    return initial_aggregation


def aggregate_responses_by_video(df, sort_order):
    initial_aggregation = df.groupby(['Type', 'URL']).agg({'Score':['count','mean','min','max'], 'TimeIndex':['first','last']})
    # initial_aggregation = initial_aggregation.reset_index()
    if sort_order == 'Similarity':
        initial_aggregation = initial_aggregation.sort_values(by=('Score', 'max'),ascending=False)
    if sort_order == 'Frame Count':
        initial_aggregation = initial_aggregation.sort_values(by=('Score', 'count'),ascending=False)
    initial_aggregation = initial_aggregation.reset_index()
    return initial_aggregation


def render_video_results(aggregated_dataframe, confidence_score,selected_indices = None):
    st.header("Search Results: Videos")
    if len(aggregated_dataframe.loc[aggregated_dataframe['Type'] == "video"]) == 0:
        st.write("No videos found")
        return
    num_of_col = 3
    i = 0
    vid_col = st.columns(num_of_col)
    for row in aggregated_dataframe.itertuples():
        if row[1] != 'video':
            continue
        index = row[0]
        show_video = selected_indices is not None and index in selected_indices
        if show_video: 
            input_type = row[1]
            input_url = row[2]
            input_score_count = row[3]
            input_score_mean = row[4]
            input_score_min = row[5]
            input_score_max = row[6]
            input_time_start = row[7]
            input_time_end = row[8]
            try:
                t = datetime.datetime.strptime(str(input_score_count), "%H:%M:%S.%f")
                delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
                total_seconds = int(delta.total_seconds())
                if input_score_max < confidence_score:
                    continue
                vid_col[index%num_of_col].video(input_url,start_time=total_seconds)
            except ValueError:
                pass
            with vid_col[i%num_of_col]:
                with st.spinner('Rendering video...'):
                    st.video(get_object_from_url(input_url, auth._pat))
                    st.write(f"Ind: {index}, Confidence: {input_score_max}, Segment counts: {str(input_score_count)}, Time: {str(input_time_start)} - {str(input_time_end)}   \n   ")
                    st.markdown("""---""")
            i += 1


def render_images(images_df, confidence_score, selected_indices = None):
    st.header("Search Results: Images")
    if len(images_df.loc[images_df['Type'] == "image"]) == 0:
        st.write("No images found")
        return
    icols = st.columns(4)
    i=0
    for row in images_df.itertuples():
        if row[1] != 'image':
            continue
        with icols[i%4]:
            if row[4] < confidence_score:
                continue
            st.image(get_object_from_url(row[2], auth._pat))
            st.write(f"Ind: {row[0]}, Confidence: {row[4]}   \n   ")
            st.markdown("""---""")
        i += 1



@st.cache_data
def convert_df(df):
   return df.to_csv().encode('utf-8')


def reset_session_state():
    st.session_state.evaluate_clicked = False


def show_pie_chart(loc, small_number, total, threshold): 
    labels = f'{small_number} Found above {threshold*100}%', f'Below {threshold*100}%'
    sizes = [small_number, total-small_number]
    colors_to_show = ["#025BF5","#DDD"]
    explode = (0.1, 0) 

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=180,colors=colors_to_show)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    ax1.set_title(f"Total inputs Searched: {total}")
    loc.pyplot(fig1)


def show_bar_chart(scores):
    st.write(f"X: Score %")
    st.write(f"Y: Count")
    z = pd.DataFrame(scores.multiply(100).round()).Score.value_counts()
    st.bar_chart(z)


def display():
  global stub, auth
  
  if 'page_number' not in st.session_state:
    st.session_state.page_number = 0

  _, header_col2, _ = st.columns(3)
  with header_col2:
    st.markdown(
        "<h1 style='text-align: center;'>Visual Face Search</h1>",
        unsafe_allow_html=True)
  
  expander = st.expander("**Input Image**",True) 
  col1, col2 = expander.columns((4, 1))
      
  with col2:
    with st.form(key="model-inputs"):
      st.subheader('Please enter minimum confidence score')
      confidence = st.slider('Confidence:', min_value=0.0, max_value=1.0, value=0.7)
      sort_order = st.radio(
                   'Please select the sort order:',
                   ('Similarity','Frame Count'),
                   horizontal=True
      )
      submitted = st.form_submit_button('Evaluate')

  if submitted:
    with col2:
        st.write("Confidence set to " + str(confidence))
        st.write("Sort order set to " + str(sort_order))
    st.session_state['evaluate_clicked'] = True    

  with col1:
    st.write('Please upload the face of someone you are searching for')

    uploaded_file = st.file_uploader("Choose an image", on_change=reset_session_state)
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        col1.image(file_bytes)
        
  if uploaded_file is not None and st.session_state['evaluate_clicked']:
    with st.spinner('Fetching results...'):
      prediction_response = find_matches(file_bytes,confidence)
      response_df = generate_dataframe(prediction_response, confidence)
      detailed_findings = detail_responses(response_df, sort_order)
      total_processed= total_processed_inputs()
    
    st.subheader("Detailed Findings")
    st.data_editor(detailed_findings, column_config={
        "URL": st.column_config.LinkColumn(
            "URL",
            validate="^https://[a-z]+\.streamlit\.app$",
            max_chars=100,
        ),
        "Input URL": st.column_config.LinkColumn(
            "Input URL",
            validate="^https://[a-z]+\.streamlit\.app$",
            max_chars=100,
        )
    }, disabled=True)
    
    csv = convert_df(detailed_findings)
    st.download_button("Download Table",csv,f"{uploaded_file.name}.csv","text/csv",key='download-csv')

    c1,c2 = st.columns([1,1])
    with c1:
        show_pie_chart(c1,len(detailed_findings.index),total_processed,confidence)
    with c2:
        show_bar_chart(detailed_findings['Score'])

    # aggregated_findings_t = aggregate_responses_by_timeindex(response_df, sort_order)
    aggregated_findings_v = aggregate_responses_by_video(response_df, sort_order)
    st.subheader("Aggregated Findings (For Videos)")
    st.data_editor(aggregated_findings_v, column_config={
        "URL": st.column_config.LinkColumn(
            "URL",
            validate="^https://[a-z]+\.streamlit\.app$",
            max_chars=100,
        ),
        "Input URL": st.column_config.LinkColumn(
            "Input URL",
            validate="^https://[a-z]+\.streamlit\.app$",
            max_chars=100,
        )
    }, disabled=True)
    csv_aggv = convert_df(aggregated_findings_v)
    st.download_button("Download Table",csv_aggv,f"{uploaded_file.name}_aggv.csv","text/csv",key='download-csv-aggv') 
    
    # Display Videos/Data 
    prev, section ,next = st.columns([1, 9, 1])
    
    if prev.button("Previous"): 
        if st.session_state.page_number != 0: 
            st.session_state.page_number -=1
    if next.button("Next"): 
        st.session_state.page_number +=1

    section.write(f"Page: {st.session_state.page_number}")
    # selected_indices = st.multiselect('Select Rows:',aggregated_findings.index)

    num_of_vids_per_page = 12
    range_to_display = range(st.session_state.page_number*num_of_vids_per_page,(st.session_state.page_number+1)*num_of_vids_per_page)
    render_images(aggregated_findings_v, confidence, range_to_display)
    render_video_results(aggregated_findings_v, confidence, range_to_display)


display()