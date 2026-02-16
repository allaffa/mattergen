# if you installed mattergen as package in the virtual environment
source .venv/bin/activate
mattergen-train data_module=mp_20 \
  '~trainer.logger' \
  'trainer.accelerator=cpu' \
  'trainer.devices=1' \
  'trainer.num_nodes=1'

# using the module directly from the source code  
python -m mattergen.scripts.run data_module=mp_20 'trainer.accelerator=cpu' 'trainer.devices=1' 'trainer.num_nodes=1' '~trainer.logger'