import os
import json

def create_jsonl_from_folders(root_folder, output_folder):
    output_file = os.path.join(output_folder, 'metadata_syn_dual.jsonl')
    with open(output_file, 'w') as jsonl_file:
        images_folder = os.path.join(root_folder, 'train/Syn_dual')
        for subdir, _, files in os.walk(images_folder):
            if len(files) < 5:
                continue
            image_path = os.path.join(subdir, 'transmission.png')
            conditioning_image_path = os.path.join(subdir, 'image.png')
            conditioning_image_path2 = os.path.join(subdir, 'image2.png')
            data_info = {
                "image": image_path,
                "text": "remove glass reflection",
                "conditioning_image": conditioning_image_path,
                "conditioning_image2": conditioning_image_path2
            }
            jsonl_file.write(json.dumps(data_info) + '\n')

    output_file = os.path.join(output_folder, 'metadata_drr_dual.jsonl')
    with open(output_file, 'w') as jsonl_file:
        # ours new data aligned
        images_folder = os.path.join(root_folder, 'train/DRR_dual')
        for subdir, _, _ in os.walk(images_folder):
            reflection_folder = os.path.join(subdir, 'reflection')
            transmission_folder = os.path.join(subdir, 'transmission')
            if os.path.exists(reflection_folder) and os.path.exists(transmission_folder):
                transmission_files = os.listdir(transmission_folder)
                reflection_files = os.listdir(reflection_folder)
                if len(transmission_files) == 1 and len(reflection_files) == 2:
                    transmission_path = os.path.join(transmission_folder, transmission_files[0])
                    reflection_path1 = os.path.join(reflection_folder, reflection_files[0])
                    reflection_path2 = os.path.join(reflection_folder, reflection_files[1])
                    data_info = {
                        "image": transmission_path,
                        "text": "remove glass reflection",
                        "conditioning_image": reflection_path1,
                        "conditioning_image2": reflection_path2
                    }
                    jsonl_file.write(json.dumps(data_info) + '\n')

    output_file = os.path.join(output_folder, 'metadata_nature_real.jsonl')
    with open(output_file, 'w') as jsonl_file:
        transmissions_folder = os.path.join(root_folder, 'train/nature/transmission_layer')
        images_folder = os.path.join(root_folder, 'train/nature/blended')
        #对transmissions里的每个文件，将其路径加入到jsonl文件中
        image_files = os.listdir(images_folder)
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            transmission_path = os.path.join(transmissions_folder, image_file)
            data_info = {
                "image": transmission_path,
                "text": "remove glass reflection",
                "conditioning_image": image_path
            }
            jsonl_file.write(json.dumps(data_info) + '\n')

        transmissions_folder = os.path.join(root_folder, 'train/real/transmission_layer')
        images_folder = os.path.join(root_folder, 'train/real/blended')
        image_files = os.listdir(images_folder)
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            transmission_path = os.path.join(transmissions_folder, image_file)
            data_info = {
                "image": transmission_path,
                "text": "remove glass reflection",
                "conditioning_image": image_path
            }
            jsonl_file.write(json.dumps(data_info) + '\n')

    output_file = os.path.join(output_folder, 'metadata_drr.jsonl')
    with open(output_file, 'w') as jsonl_file:
        transmissions_folder = os.path.join(root_folder, 'train/DRR/transmission_layer')
        images_folder = os.path.join(root_folder, 'train/DRR/blended')
        #对transmissions里的每个文件，将其路径加入到jsonl文件中
        image_files = os.listdir(images_folder)
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            transmission_path = os.path.join(transmissions_folder, image_file)
            data_info = {
                "image": transmission_path,
                "text": "remove glass reflection",
                "conditioning_image": image_path
            }
            jsonl_file.write(json.dumps(data_info) + '\n')

    output_file = os.path.join(output_folder, 'metadata_syn.jsonl')
    with open(output_file, 'w') as jsonl_file:
        transmissions_folder = os.path.join(root_folder, 'train/Syn/transmission_layer')
        images_folder = os.path.join(root_folder, 'train/Syn/blended')
        #对transmissions里的每个文件，将其路径加入到jsonl文件中
        image_files = os.listdir(images_folder)
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            transmission_path = os.path.join(transmissions_folder, image_file)
            data_info = {
                "image": transmission_path,
                "text": "remove glass reflection",
                "conditioning_image": image_path
            }
            jsonl_file.write(json.dumps(data_info) + '\n')

    output_file = os.path.join(output_folder, 'metadata_test_drr_c.jsonl')
    with open(output_file, 'w') as jsonl_file:
        transmissions_folder = os.path.join(root_folder, 'test/DRR-C/transmission_layer')
        images_folder = os.path.join(root_folder, 'test/DRR-C/blended')
        #对transmissions里的每个文件，将其路径加入到jsonl文件中
        image_files = os.listdir(images_folder)
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            transmission_path = os.path.join(transmissions_folder, image_file)
            data_info = {
                "image": transmission_path,
                "text": "remove glass reflection",
                "conditioning_image": image_path
            }
            jsonl_file.write(json.dumps(data_info) + '\n')

    output_file = os.path.join(output_folder, 'metadata_test_drr_s.jsonl')
    with open(output_file, 'w') as jsonl_file:
        transmissions_folder = os.path.join(root_folder, 'test/DRR-S/transmission_layer')
        images_folder = os.path.join(root_folder, 'test/DRR-S/blended')
        #对transmissions里的每个文件，将其路径加入到jsonl文件中
        image_files = os.listdir(images_folder)
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            transmission_path = os.path.join(transmissions_folder, image_file)
            data_info = {
                "image": transmission_path,
                "text": "remove glass reflection",
                "conditioning_image": image_path
            }
            jsonl_file.write(json.dumps(data_info) + '\n')

    output_file = os.path.join(output_folder, 'metadata_test_nature.jsonl')
    with open(output_file, 'w') as jsonl_file:
        transmissions_folder = os.path.join(root_folder, 'test/Nature/transmission_layer')
        images_folder = os.path.join(root_folder, 'test/Nature/blended')
        #对transmissions里的每个文件，将其路径加入到jsonl文件中
        image_files = os.listdir(images_folder)
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            transmission_path = os.path.join(transmissions_folder, image_file)
            data_info = {
                "image": transmission_path,
                "text": "remove glass reflection",
                "conditioning_image": image_path
            }
            jsonl_file.write(json.dumps(data_info) + '\n')


    output_file = os.path.join(output_folder, 'metadata_test_real.jsonl')
    with open(output_file, 'w') as jsonl_file:
        transmissions_folder = os.path.join(root_folder, 'test/real20_420/transmission_layer')
        images_folder = os.path.join(root_folder, 'test/real20_420/blended')
        #对transmissions里的每个文件，将其路径加入到jsonl文件中
        image_files = os.listdir(images_folder)
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            transmission_path = os.path.join(transmissions_folder, image_file)
            data_info = {
                "image": transmission_path,
                "text": "remove glass reflection",
                "conditioning_image": image_path
            }
            jsonl_file.write(json.dumps(data_info) + '\n')


    output_file = os.path.join(output_folder, 'metadata_test_sir_postcard.jsonl')
    with open(output_file, 'w') as jsonl_file:
        transmissions_folder = os.path.join(root_folder, 'test/SIR2/PostcardDataset/transmission_layer')
        images_folder = os.path.join(root_folder, 'test/SIR2/PostcardDataset/blended')
        #对transmissions里的每个文件，将其路径加入到jsonl文件中
        image_files = os.listdir(images_folder)
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            transmission_path = os.path.join(transmissions_folder, image_file)
            data_info = {
                "image": transmission_path,
                "text": "remove glass reflection",
                "conditioning_image": image_path
            }
            jsonl_file.write(json.dumps(data_info) + '\n')

    output_file = os.path.join(output_folder, 'metadata_test_sir_object.jsonl')
    with open(output_file, 'w') as jsonl_file:
        transmissions_folder = os.path.join(root_folder, 'test/SIR2/SolidObjectDataset/transmission_layer')
        images_folder = os.path.join(root_folder, 'test/SIR2/SolidObjectDataset/blended')
        #对transmissions里的每个文件，将其路径加入到jsonl文件中
        image_files = os.listdir(images_folder)
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            transmission_path = os.path.join(transmissions_folder, image_file)
            data_info = {
                "image": transmission_path,
                "text": "remove glass reflection",
                "conditioning_image": image_path
            }
            jsonl_file.write(json.dumps(data_info) + '\n')

    output_file = os.path.join(output_folder, 'metadata_test_sir_wild.jsonl')
    with open(output_file, 'w') as jsonl_file:
        transmissions_folder = os.path.join(root_folder, 'test/SIR2/WildSceneDataset/transmission_layer')
        images_folder = os.path.join(root_folder, 'test/SIR2/WildSceneDataset/blended')
        #对transmissions里的每个文件，将其路径加入到jsonl文件中
        image_files = os.listdir(images_folder)
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            transmission_path = os.path.join(transmissions_folder, image_file)
            data_info = {
                "image": transmission_path,
                "text": "remove glass reflection",
                "conditioning_image": image_path
            }
            jsonl_file.write(json.dumps(data_info) + '\n')

if __name__ == "__main__":
    root_folder = '/data_hdd2/users/yangchen/projects/SIRR/DRR'
    output_folder = '/data_hdd2/users/yangchen/projects/SIRR/DRR_jsonl'
    create_jsonl_from_folders(root_folder, output_folder)