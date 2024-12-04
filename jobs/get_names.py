# grabs all the names of all files including the campaign and specifier thing
# used for the condor scripts

from os import listdir, path



NOMINAL_PATH = "/eos/home-c/cahampso/ntuples/nominal"
CAMPAIGNS = ["mc20a", "mc20d", "mc20e"]
IDENTIFIERS = ["p5855", "p6266"]

all_files = []


for campaign in CAMPAIGNS:
    for identifier in IDENTIFIERS:
        ntuple_files = listdir(path.join(NOMINAL_PATH, campaign, identifier))
        for ntuple_file in ntuple_files:
            all_files.append(campaign + "/" + identifier + "/" + ntuple_file)


# strip the trailing space and comma from the final entry
# probably not necessary but who cares
# all_files[-1] = all_files[-1][:-2]
all_files_str = "\n".join(all_files)
#print("infiles = " + all_files_str)

OUTPUT_FILE_PATH = "./names.txt"
with open(OUTPUT_FILE_PATH, "w") as output_file:
    output_file.write(all_files_str)

    
