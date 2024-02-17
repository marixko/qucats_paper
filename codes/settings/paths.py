import os


mycwd = os.path.abspath(os.getcwd())

main_dir_name = 'qucats_paper'

if main_dir_name in mycwd or os.path.isdir(os.path.join(mycwd, main_dir_name)):
    main_dir = os.path.join(mycwd.split(main_dir_name)[0], main_dir_name)
else:
    raise ImportError(f'Current directory does not contain {main_dir_name}.')

codes_path = os.path.join(main_dir, "codes")
dust_path = os.path.join(codes_path, "utils", "dustmaps")


img_path = os.path.join(main_dir,  'img')
if not os.path.exists(img_path):
    os.makedirs(img_path)


data_path = os.path.join(main_dir, 'data')
if not os.path.exists(data_path):
    os.makedirs(data_path)

match_path = os.path.join(data_path, 'crossmatch')
if not os.path.exists(match_path):
    os.makedirs(match_path)

validation_path = os.path.join(data_path, 'crossvalidation')
if not os.path.exists(validation_path):
    os.makedirs(validation_path)


result_path = os.path.join(main_dir,  'results')
if not os.path.exists(result_path):
    os.makedirs(result_path)

rf_path = os.path.join(result_path, 'rf')
if not os.path.exists(rf_path):
    os.makedirs(rf_path)

flex_path = os.path.join(result_path, 'flexcode')
if not os.path.exists(flex_path):
    os.makedirs(flex_path)

bmdn_path = os.path.join(result_path, 'bmdn')
if not os.path.exists(bmdn_path):
    os.makedirs(bmdn_path)
