# Copyright 2023 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START storageImports]
# [START storageAdditionalImports]
import io
import pathlib

# from PIL import Image

from firebase_admin import initialize_app

# initialize_app()
from firebase_admin import storage
# [END storageAdditionalImports]

# [START storageSDKImport]
from firebase_functions import storage_fn
# [END storageSDKImport]
# [END storageImports]


# [START storageGenerateThumbnail]
# [START storageGenerateThumbnailTrigger]
@storage_fn.on_object_finalized()
def generatethumbnail(event: storage_fn.CloudEvent[storage_fn.StorageObjectData]):
    """When an image is uploaded in the Storage bucket, generate a thumbnail
    automatically using Pillow."""
# [END storageGenerateThumbnailTrigger]

    # [START storageEventAttributes]
    
    # [END storageThumbnailGeneration]
# [END storageGenerateThumbnail]