import os
import json
import subprocess
import sys
import re


class ConfigLoader:
    def __init__(self, project_base_dir):
        self.project_base_dir = project_base_dir
        self.python_executable = self._find_python_executable()
        self.cache_path = os.path.join(self.project_base_dir, "MangaStudio_Data", "temp", "schema_cache.json")

        self.backend_schema = self._load_backend_schema()
        self.ui_map = self._load_ui_map()
        self.tasks_config = self._load_tasks_config() # This line is correct

        if not self.backend_schema:
            raise RuntimeError("Failed to load backend configuration schema.")

        # The data is built and stored directly as attributes, not through getter methods
        self.factory_defaults = self._parse_factory_defaults()
        self.full_config_data = self._build_full_config_data()

    def _find_python_executable(self):
        venv_path_win = os.path.join(self.project_base_dir, 'venv', 'Scripts', 'python.exe')
        venv_path_unix = os.path.join(self.project_base_dir, 'venv', 'bin', 'python')
        if os.path.exists(venv_path_win):
            return venv_path_win
        elif os.path.exists(venv_path_unix):
            return venv_path_unix
        return sys.executable

    def _load_backend_schema(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    print("[ConfigLoader] Loading schema from cache...")
                    return json.load(f)
            except Exception:
                pass

        print("[ConfigLoader] Fetching fresh configuration schema...")
        try:
            command = [self.python_executable, "-m", "manga_translator", "config-help"]
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=True)
            schema_data = self._parse_schema_output(result.stdout)
            if schema_data is None:
                raise ValueError("Schema command did not return valid JSON.")
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(schema_data, f, indent=4)
            return schema_data
        except Exception as e:
            print(f"[ERROR] Could not fetch schema: {e}")
            return None
    
    def _parse_schema_output(self, stdout):
        """Extracts the JSON portion of the schema output."""
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            cleaned_stdout = self._strip_ansi(stdout)
            json_start = cleaned_stdout.find('{')
            json_end = cleaned_stdout.rfind('}')
            if json_start == -1 or json_end == -1 or json_end < json_start:
                return None
            try:
                return json.loads(cleaned_stdout[json_start:json_end + 1])
            except json.JSONDecodeError:
                return None

    def _strip_ansi(self, text):
        ansi_escape = re.compile(r'\x1B\[[0-9;]*[A-Za-z]')
        return ansi_escape.sub('', text)

    def _load_ui_map(self):
        map_path = os.path.join(self.project_base_dir, 'MangaStudio_Data', 'ui_map.json')
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] UI map loading failed: {e}")
            return {}

    def _load_tasks_config(self):
        """Loads the special tasks configuration from tasks.json."""
        tasks_path = os.path.join(self.project_base_dir, 'MangaStudio_Data', 'tasks.json')
        try:
            with open(tasks_path, 'r', encoding='utf-8') as f:
                print("[ConfigLoader] Loading tasks configuration...")
                return json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] tasks.json not found at: {tasks_path}")
            return {}
        except Exception as e:
            print(f"[ERROR] Tasks config loading failed: {e}")
            return {}
    
    def _get_definition_from_ref(self, ref_path):
        try:
            parts = ref_path.split('/')[1:]
            node = self.backend_schema
            for part in parts:
                node = node[part]
            return node
        except Exception:
            return None

    def _parse_factory_defaults(self):
        """Deep-parses the schema to get ALL default values, including nested ones."""
        if not self.backend_schema:
            return {}
        defaults = {}
        properties = self.backend_schema.get("properties", {})

        for prop_key, prop_value in properties.items():
            # This handles nested defaults (e.g., from 'detector', 'render')
            if "default" in prop_value and isinstance(prop_value.get("default"), dict):
                defaults.update(prop_value["default"])
            # This handles simple root-level properties (e.g., 'kernel_size')
            elif "default" in prop_value:
                defaults[prop_key] = prop_value["default"]
        return defaults

    def _build_full_config_data(self):
        """Builds the final, merged config data for the UI, reading ALL properties."""
        if not self.ui_map:
            return {}
        full_data = {}
        all_properties = {}

        # 1. Gather all root-level properties
        root_props = self.backend_schema.get("properties", {})
        all_properties.update(root_props)

        # 2. Gather all nested properties from complex types (e.g., DetectorConfig)
        for prop in root_props.values():
            ref_path = prop.get("allOf", [{}])[0].get('$ref')
            if ref_path:
                config_def = self._get_definition_from_ref(ref_path)
                if config_def and "properties" in config_def:
                    all_properties.update(config_def["properties"])

        # 3. Build the final data structure using the UI map as the guide
        for key, ui_info in self.ui_map.items():
            if key.startswith("__"):
                continue
            merged_info = ui_info.copy()
            merged_info['key'] = key

            # Use the already parsed factory default
            merged_info['default'] = self.factory_defaults.get(key)

            # Add enum values (for dropdowns) if they exist
            prop_def = all_properties.get(key)
            if prop_def and isinstance(prop_def, dict):
                ref_path = prop_def.get("allOf", [{}])[0].get('$ref')
                if ref_path:
                    enum_def = self._get_definition_from_ref(ref_path)
                    if enum_def and "enum" in enum_def:
                        merged_info['values'] = enum_def["enum"]
            full_data[key] = merged_info
        return full_data

    def get_tasks_config(self):
        """Returns the loaded tasks configuration."""
        return self.tasks_config

    def get_factory_defaults(self):
        return self.factory_defaults

    def get_tab_order(self):
        return self.ui_map.get("__tab_order__", [])
