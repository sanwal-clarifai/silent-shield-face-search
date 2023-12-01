import os
from PIL import Image
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
from PIL import Image
import io

auth = ClarifaiAuthHelper.from_streamlit(st)
print(auth._pat)
stub = auth.get_stub()
userDataObject = auth.get_user_app_id_proto()
userDataClarifaiMain= resources_pb2.UserAppIDSet(user_id='clarifai', app_id='main')
print(userDataObject)
