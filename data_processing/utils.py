import yaml
from pathlib import Path
import autorootcwd
import re

def read_yaml(filepath: str):
    stream = open(filepath, 'r')
    return yaml.safe_load(stream)

class TRExFitter:
    def parse_trexfitter_config(self, filepath: str):
        lines = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.replace("\n", "")
                if len(line) < 5 or line[0] == "#":
                    continue
                if line.startswith("INCLUDE"):
                    tokens = line.split(": ")
                    filepath_tokens = tokens[1].split("/")
                    if len(filepath_tokens) < 1:
                        continue
                    include_filepath_raw = Path()
                    include_filepath = include_filepath_raw.joinpath("trex-fitter", "configs", *filepath_tokens).resolve()
                    
                    if not include_filepath.exists():
                        print(f"[ERROR] parse_trexfitter_config(): failed to find file:\n\t{include_filepath}")
                        exit(1)
                    
                    with include_filepath.open() as include_file:
                        for l in include_file:
                            l = l.replace("\n", "")
                            if len(l) < 5 or l[0] == "#":
                                continue
                            lines.append(l)
                    continue
                        
                lines.append(line)

        res = {}
        parsing_block = False
        res_ = {}
        main_key = ""
        sub_key = ""
        whitespace = re.compile(r"^\s+.*")
        for i,line in enumerate(lines):
            tokens = line.split(": ")
            if len(tokens) != 2:
                continue
            # print(f"({i}/{len(lines)})->{line}")
            match_res = re.match(whitespace, line)
            # if we find a starting block and we are not currently parsing one,
            # this we start parsing the block
            if (match_res is None) and (not parsing_block):
                # print("  starting block, beginning parsing")
                parsing_block = True
                main_key = tokens[0]
                sub_key = tokens[1].replace("\"", "")
                res_ = {}
            # if we find sub-block and we are parsing a block,
            # add it to the current block
            elif (match_res is not None) and parsing_block:
                # print("  parsing sub-block entry")
                fixed_key, fixed_val = tokens[0].strip(), tokens[1].strip()
                res_[fixed_key] = fixed_val
            # if we find a starting block and we are currently parsing one,
            # stop parsing the block and add the entire sub-contents
            # then read in the new keys
            elif (match_res is None) and parsing_block:
                # print("  found another block. appending previous, starting another")
                if main_key in res.keys():
                    res[main_key].append([sub_key, res_])
                else:
                    res[main_key] = [[sub_key, res_]]
                main_key = tokens[0]
                sub_key = tokens[1].replace("\"", "")
                res_ = {}
                
            # the above cases don't consider the final block of data
            # because we only add the block if we find another while we are parsing
            # that is also why this is a separate if, not another elif branch
            # because we need to parse the last line of the block before doing this
            if (i+1) >= len(lines):
                # print("  finished last block, appending this")
                if main_key in res.keys():
                    res[main_key].append([sub_key, res_])
                else:
                    res[main_key] = [[sub_key, res_]]
                main_key = tokens[0]
                sub_key = tokens[1].replace("\"", "")
                res_ = {}
                
        self.config_data = res

    def parse_trexfitter_replacement(self, filepath: str):
        lines = []
        with open(filepath, 'r') as f:
            for line in f:
                if line[0] == "#" or len(line) < 3:
                    continue
                lines.append(line)
        res = {}
        for line in lines:
            tokens = line.split(": ")
            if len(tokens) != 2:
                continue;
            tokens[1] = tokens[1].replace("\n", "")
            res[tokens[0]] = tokens[1]
        self.replacement_data = res

    def get_weight_expr(self):
        weight_expr = None
        for job in self.config_data["Job"]:
            if "MCweight" not in job[1].keys():
                continue
            weight_expr = job[1]["MCweight"].replace("\"", "")
            for k,v in self.replacement_data.items():
                if k in weight_expr:
                    weight_expr = weight_expr.replace(k, v)
            break
        if weight_expr is None:
            print("Failed to find an MCweight field in the Job block. Using a weight of 1...")
        return weight_expr

    def get_selection_expr(self, config_data: dict):
        region = config_data["Region"]
        selection_expr = None
        for reg in self.config_data["Region"]:
            if reg[0] == region:
                if "Selection" not in reg[1].keys():
                    continue
                selection_expr = reg[1]["Selection"].replace("\"", "")
                for k,v in self.replacement_data.items():
                    print(f"testing for {k} in {selection_expr}")
                    if k in selection_expr:
                        selection_expr = selection_expr.replace(k, v)
                break
        if selection_expr == "":
            print(f"Failed to find a Selection field in the region block '{region}'. Using a selection of 1...")

        return selection_expr
    
    def __init__(self, config_file_path: str):
        self.parse_trexfitter_config(config_file_path)
        for job in self.config_data["Job"]:
            replacement_file_path_tokens = job[1]["ReplacementFile"].split("/")
            replacement_file_path = Path()
            replacement_file_path = replacement_file_path.joinpath("trex-fitter", *replacement_file_path_tokens).resolve()
            self.parse_trexfitter_replacement(replacement_file_path)
            break


    def print(self):
        for k,v in self.config_data.items():
            print(f"========== BEGIN {k} BLOCKS ==========")
            for block in v:
                print(f"{block[0]}:")
                for block_k,block_v in block[1].items():
                    print(f"  - {block_k}: {block_v}")
            print(f"========== END {k} BLOCKS ==========")

    def get_block(self, block_type: str, block_name=None, key=None) -> dict:
        if block_type not in self.config_data.keys():
            print(f"Block type '{block_type}' not found in the trex-fitter config.")
            exit(1)

        block = {}
        if block_name is None:
            for blocks in self.config_data[block_type]:
                block = blocks[1]

        for blocks in self.config_data[block_type]:
            if blocks[0] == block_name:
                block = blocks[1]

        if block is None:
            print(f"Failed to find block '{block_name}' in '{block_type}' list.");
            exit(1)

        if key is None:
            return block

        if key in block.keys():
            return block[key]

        print(f"Failed to find key '{key}' in block '{block_name}'")
        exit(1)

    def replace(self, replacement_string: str) -> str:
        for k,v in self.replacement_data.items():
            #print(f"testing for {k} in {replacement_string}")
            if k in replacement_string:
                replacement_string = replacement_string.replace(k, v)
        return replacement_string

        
        
                
