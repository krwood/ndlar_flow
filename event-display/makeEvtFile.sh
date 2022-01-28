#!/bash/bin/

#python to_evd_file.py --in_filename /global/homes/k/kwood/dunend/software/module0_flow_tutorial/tutorial/module0_flow/larndsim.example.out.h5 --out_filename larndsim.example.out.evd.h5 --geometry_file multi_tile_layout-2.2.16.yaml
python module0_evd.py --filename /global/homes/k/kwood/dunend/software/module0_flow_tutorial/tutorial/module0_flow/larndsim.example.out.h5 --geometry_file multi_tile_layout-2.2.16.yaml
