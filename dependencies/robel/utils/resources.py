# Copyright 2019 The ROBEL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper methods to locate asset files."""

import collections
import logging
import os
import shutil
import tempfile
from typing import Callable, Dict, Optional

_MODULE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')

_SCENES_DIR_COMPONENT = 'robel'


def get_asset_path(path: str):
    """Returns the absolute path of the given fully-qualified resource path.

    Example:
        >>> get_asset_path('robel/dclaw/assets/')

    Args:
        path: The path to the resource, with components separated by slashes.
    """
    if path.startswith('robel-scenes'):
        asset_path = os.path.join(_MODULE_DIR, _SCENES_DIR_COMPONENT, path)
    elif path.startswith('robel'):
        asset_path = os.path.join(_MODULE_DIR, path)
    else:
        raise ValueError('Unknown path root: ' + path)
    asset_path = os.path.normpath(asset_path)
    return asset_path


def get_resource(path: str, mode: str = 'rb'):
    """Returns the contents of the given resource file path."""
    with open(path, mode=mode) as f:
        return f.read()


class AssetBundle:
    """Represents a bundle of assets files"""

    def __init__(self,
                 dest_path: Optional[str] = None,
                 resource_fn: Callable[[str], bytes] = get_resource,
                 dry_run: bool = False,
                 verbose: bool = False):
        """Creates a new asset bundle.

        Args:
            dest_path: The destination directory to copy the bundle to.
            resource_fn: The function used to get the contents of the file.
            dry_run: If True, does not write files to the destination.
            verbose: If True, logs copied files.
        """
        self._resource_fn = resource_fn
        self._dry_run = dry_run
        self._verbose = verbose
        self._copied_resources = collections.OrderedDict()
        self._needs_cleanup = False

        if dest_path is None and not dry_run:
            dest_path = tempfile.mkdtemp()
            self._needs_cleanup = True
        self._dest_path = dest_path or ''

    @property
    def copied_paths(self) -> Dict[str, str]:
        """Returns the copied resource paths."""
        return self._copied_resources

    def cleanup(self):
        """Removes the temporary directory."""
        if self._needs_cleanup and self._dest_path:
            shutil.rmtree(self._dest_path)
            self._needs_cleanup = False

    def add_mujoco(self, main_path: str) -> str:
        """Adds the given MuJoCo XML file to the bundle."""
        from xml.etree import ElementTree as etree
        main_path = os.path.normpath(main_path)
        main_dir = os.path.dirname(main_path)
        directory_context = {
            'mesh': main_dir,
            'texture': main_dir,
        }

        # Traverse the XML tree depth-first.
        node_stack = []
        node_stack.append((directory_context, main_path))
        while node_stack:
            directories, file_path = node_stack.pop()
            base_dir = os.path.dirname(file_path)

            xml_contents = self._copy_asset(file_path)
            node = etree.fromstring(xml_contents)
            children = []

            # Update the directories if a compiler tag is present.
            for child in node.iter('compiler'):
                if 'meshdir' in child.attrib:
                    directories['mesh'] = os.path.join(base_dir,
                                                       child.attrib['meshdir'])
                if 'texturedir' in child.attrib:
                    directories['texture'] = os.path.join(
                        base_dir, child.attrib['texturedir'])

            for child in node.iter():
                # Resolve mesh and texture children with file tags.
                if child.tag in directories:
                    if 'file' in child.attrib:
                        asset_path = os.path.join(directories[child.tag],
                                                  child.attrib['file'])
                        if asset_path not in self._copied_resources:
                            self._copy_asset(asset_path)
                # Traverse includes.
                elif child.tag == 'include':
                    child_path = os.path.join(base_dir, child.attrib['file'])
                    children.append((directories.copy(), child_path))

            # Traverse children in visit order.
            node_stack.extend(reversed(children))

        return self._copied_resources[main_path]

    def _copy_asset(self, asset_path: str) -> bytes:
        """Copies an asset and returns its contents."""
        assert not asset_path.startswith('/'), asset_path
        asset_path = os.path.normpath(asset_path)
        if self._verbose:
            logging.info('Found asset: %s', asset_path)
        contents = self._resource_fn(asset_path)

        # Copy the asset to the destination.
        if asset_path not in self._copied_resources:
            copy_path = os.path.join(self._dest_path, asset_path)
            if not self._dry_run:
                self._write_asset(copy_path, contents)
            self._copied_resources[asset_path] = copy_path

        return contents

    def _write_asset(self, write_path: str, contents: bytes):
        """Writes the contents to the given path."""
        copy_dir = os.path.dirname(write_path)
        if not os.path.isdir(copy_dir):
            os.makedirs(copy_dir)
        with open(write_path, 'wb') as f:
            f.write(contents)

    def __enter__(self):
        """Enables use as a context manager."""
        return self

    def __exit__(self, *args):
        """Enables use as a context manager."""
        self.cleanup()
