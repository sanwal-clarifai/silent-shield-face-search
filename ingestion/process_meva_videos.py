import os
import subprocess
import sys
## Import in the Clarifai gRPC based objects needed
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_pb2, status_code_pb2
from google.protobuf.struct_pb2 import Struct
from google.protobuf.json_format import MessageToDict


auth_key = "Key " + os.getenv('PAT')
app_owner = "kirkdebaets"
app_id = "nyt-006"
max_size = 70
channel = ClarifaiChannel.get_grpc_channel()
# channel = ClarifaiChannel.get_json_channel()
stub = service_pb2_grpc.V2Stub(channel)
metadata = (('authorization', auth_key),)
userDataObject = resources_pb2.UserAppIDSet(user_id=app_owner, app_id=app_id)
input_metadata = Struct()

def split_video(filename):
    # change it to an mp4
    # mp4box -add 2018-03-05.11-30-01.11-35-01.bus.G331.r13.avi -new 2018-03-05.11-30-01.11-35-01.bus.G331.r13.mp4
    mp4name = filename.replace('.avi','.mp4')
    print(mp4name)
    print(subprocess.list2cmdline(['/opt/homebrew/bin/mp4box',
                             'add', filename, mp4name]))
    format_results = subprocess.run(['/opt/homebrew/bin/mp4box',
                                     '-add', filename, mp4name])
    # chunk it down to ~1 min segments
    mp4box_results = subprocess.run(['/opt/homebrew/bin/mp4box',
                                     '-v', '-split', '60', mp4name ],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
    # print('exit code was: %d' % mp4box_results.returncode)
    details = str(mp4box_results.stderr)
    split_count = details.count('Box free')
    return split_count, mp4name

def parse_filename(filename):
    separator='-'
    # ['2018-03-11', '23-55-00', '00-00-00', 'bus', 'G331', 'r13', 'mp4']
    [start_date, start_time, end_time, location, camera, junk1, junk2] = filename.split('.')
    input_id = separator.join([location, camera, start_date, start_time])
    # print(start_date, start_time, location, camera)
    # print(input_id)
    return input_id

def upload_video(target_file, input_id):
    print('Target file: ' + str(target_file))
    print('Input ID: ' + str(input_id))

    with open(target_file, 'rb') as img_file:
        file_bytes = img_file.read()

    post_inputs_response = stub.PostInputs(
        service_pb2.PostInputsRequest(
            user_app_id=userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
            inputs=[
                resources_pb2.Input(
                    id=input_id,
                    data=resources_pb2.Data(
                        video=resources_pb2.Video(
                            base64=file_bytes
                        ),
                        metadata = input_metadata
                    )
                )
            ]
        ),
        metadata=metadata
    )
    #
    post_inputs_dict = MessageToDict(post_inputs_response)
    req_id = post_inputs_dict['status']['reqId']
    print(req_id)
    if post_inputs_response.status.code != status_code_pb2.SUCCESS:
        print("There was an error with your request!")
        print(post_inputs_response)

def process_oversized(filename, input_id):
    num_splits, mp4name = split_video(filename)
    for split in range(1, num_splits+1):
        suffix = "_00"+str(split)+".mp4"
        next_file = mp4name.replace('.mp4',suffix)
        # print(next_file)
        id_size = len(input_id)
        next_id = input_id[:-1] + str(split)
        # print(next_id)
        upload_video(next_file, next_id)

def main():
    file = sys.argv[1]
    # file = '/Users/kirkdebaets/projects/customers/nyt/Data/mevadata/bus_331/2018-03-11.23-55-00.00-00-00.bus.G331.r13.avi'
    # file = '/Users/kirkdebaets/projects/customers/nyt/Data/mevadata/bus_331/2018-03-15.15-55-00.16-00-00.bus.G331.r13.avi'
    vid_filename = os.path.basename(file)
    vid_size = os.path.getsize(file)/1024/1024

    print(vid_size, vid_filename)
    input_id = parse_filename(vid_filename)
    input_metadata.update({"filename":vid_filename})
    if vid_size > max_size:
        process_oversized(file, input_id)
    else:
        upload_video(file, input_id)


if __name__ == "__main__":
    main()
