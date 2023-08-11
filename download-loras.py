'''
DEF CON 31 AI Village - LLMs: Loose Lips Multipliers
Kyle Easterly & Mitch Kitter
https://www.purpleaerospace.com
download-loras.py: LoRA bulk download script
'''

from huggingface_hub import snapshot_download

v1_80_option = '1) Version 1, 80 synthetic proprietary interactions [1.78GB]'
v1_200_option = '2) Version 2, 200 synthetic proprietary interactions [3.00GB]'
v1_300_option = '3) Version 2, 300 synthetic proprietary interactions [2.43GB]'

v1_80_prefix = 'kyleeasterly/openllama-7b_purple-aerospace-v1-80-'
v2_200_prefix = 'kyleeasterly/openllama-7b_purple-aerospace-v2-200-'
v2_300_prefix = 'kyleeasterly/openllama-7b_purple-aerospace-v2-300-'

v1_80_ratios = ['0', '1', '2', '4', '8', '10', '12', '14', '16', '18', '20', '22', '24', '26', '28', '30', '32', '64', '432']
v1_200_ratios = ['0', '1', '2', '4', '6', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '20', '22', '24', '26', '28', '30', '32', '48', '64', '72', '80', '88', '96', '102', '128', '173']
v1_300_ratios = ['0', '1', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '24', '26', '28', '30', '32', '48', '64', '72', '80', '88', '96', '104', '115']

options = {
    "1": (v1_80_prefix, v1_80_ratios),
    "2": (v2_200_prefix, v1_200_ratios),
    "3": (v2_300_prefix, v1_300_ratios)
}

def get_user_input():
    while True:
        print(v1_80_option)
        print(v1_200_option)
        print(v1_300_option)
        user_input = input("Please select an option (1, 2, or 3): ")
        
        if user_input in options:
            return options[user_input]
        else:
            print("Invalid input. Please try again.")

selected_prefix, selected_ratios = get_user_input()

for ratio in selected_ratios:
    print("Downloading " + selected_prefix + ratio + "..." + str(selected_ratios.index(ratio) / len(selected_ratios) * 100) + "% complete")
    snapshot_download(repo_id = selected_prefix + ratio, repo_type = 'model')
