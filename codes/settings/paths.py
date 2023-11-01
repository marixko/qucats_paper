import os

#%%
mycwd = os.path.abspath(os.getcwd())
if mycwd.split(os.sep)[-1] == "qucats_paper":
    parent_cwd = mycwd
    codes_path = os.path.join(parent_cwd,"codes")
    pass

else:
    os.chdir('..')
    codes_path = os.path.abspath(os.getcwd())
    os.chdir('..')
    parent_cwd = os.path.abspath(os.getcwd())
    os.chdir(mycwd)
#%%



data_path = os.path.join(parent_cwd, 'data')
if not os.path.exists(data_path):
    os.makedirs(data_path)

result_path = os.path.join(parent_cwd,  'results')
if not os.path.exists(result_path):
    os.makedirs(result_path)

img_path = os.path.join(parent_cwd,  '_img')
if not os.path.exists(img_path):
    os.makedirs(img_path)

dust_path = os.path.join(codes_path, "utils", "dustmaps")

# -----
match_path = os.path.join(data_path, 'crossmatch')
if not os.path.exists(match_path):
    os.makedirs(match_path)

validation_path = os.path.join(data_path, 'validation')
if not os.path.exists(validation_path):
    os.makedirs(validation_path)


# ------

rf_path = os.path.join(result_path, 'validation' ,'rf')
if not os.path.exists(rf_path):
    os.makedirs(rf_path)

flex_path = os.path.join(result_path,'validation', 'flexcode')
if not os.path.exists(flex_path):
    os.makedirs(flex_path)

bnn_path = os.path.join(result_path,'validation', 'bnn')
if not os.path.exists(bnn_path):
    os.makedirs(bnn_path)


