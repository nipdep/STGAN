#! usr/bin/python3
#%%
import os
import Communicator

print(os.getcwd())
#%%
cmm = Communicator.Communicator('../../logs/urls.txt', '../../')
# cmm.uploadFile('req', '../../requirements.txt')
# cmm.uploadFolder('src', '../../src')
#%%
cmm.downloadFile('req', '../')

# %%
