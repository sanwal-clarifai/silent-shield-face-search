from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
import urllib.request
import streamlit as st
import requests


def init_session_state(st, auth):
  stub = auth.get_stub()
  metadata = auth.metadata
  userDataObject = auth.get_user_app_id_proto()

  page_size = 3

  # Shared API calls before any page renders. This can be used to save state for a particular browser
  # session. I'm using this here to show an example of how when the app loads we can get a value like
  # the counts of inputs, and then use it within the various pages of the multi-page app.
  # It's not clear when the session state ever gets cleared or where it is stored.
  if 'get_input_count_response' not in st.session_state:
    get_input_count_response = stub.GetInputCount(
        service_pb2.GetInputCountRequest(user_app_id=userDataObject), metadata=metadata)
    if get_input_count_response.status.code != status_code_pb2.SUCCESS:
      raise Exception("Get input count failed, response: %r" % get_input_count_response)
    counts = get_input_count_response.counts
    print(counts)
    total = get_input_count_response.counts.processed + get_input_count_response.counts.errors + get_input_count_response.counts.errors + get_input_count_response.counts.to_process
    print(total)

    st.session_state['counts'] = counts
    st.session_state['total'] = total
    st.session_state['get_input_count_response'] = get_input_count_response


def concept_key(concept):
  return "%s (%s)" % (concept.id, concept.name)


def get_annotations_for_input_batch(stub, userDataObject, metadata, inputs):

  annotations = []
  if len(inputs) == 0:
    return annotations
  input_ids = [inp.id for inp in inputs]

  list_annotations_response = stub.ListAnnotations(
      service_pb2.ListAnnotationsRequest(
          user_app_id=
          userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
          input_ids=input_ids,
          per_page=1000,  # FIXME(zeiler): this needs proper pagination.
      ),
      metadata=metadata)
  if list_annotations_response.status.code != status_code_pb2.SUCCESS:
    print("There was an error with your request!")
    print("\tCode: {}".format(list_annotations_response.outputs[0].status.code))
    print("\tDescription: {}".format(list_annotations_response.outputs[0].status.description))
    print("\tDetails: {}".format(list_annotations_response.outputs[0].status.details))
    raise Exception("List annotations failed, status: " +
                    list_annotations_response.status.description)
  for annotation_object in list_annotations_response.annotations:
    if len(annotation_object.data.concepts) > 0 or len(annotation_object.data.regions) > 0:
      # print(annotation_object)
      annotations.append(annotation_object)
    else:
      print("ZZZZZZZZZZZZZZZZZZ")
      print(annotation_object)

  return annotations


def predict_from_image(stub, metadata, img_bytes, user_id, app_id, model_id, version_id):
  '''
      '''
  request = service_pb2.PostModelOutputsRequest(
      user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
      # This is the model ID of a publicly available General model. You may use any other public or custom model ID.
      model_id=model_id,
      inputs=[
          resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(base64=img_bytes)))
      ])
  if version_id is not None:
    request.version_id = version_id

  response = stub.PostModelOutputs(request, metadata=metadata)
  # print(response)
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostModelOutputs request failed: %r" % response)

  return response

@st.cache_data
def get_object_from_url(url, pat):
  hdr = f"Bearer {pat}"
  headers = {"Authorization": hdr}
  response = requests.get(url, headers=headers)
  return response.content