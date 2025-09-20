# python perturb.py --adata_path ../../../Data/King/Simulated_adatas/adata_Sham_1_center_345.h5ad --adata_save_path ../../../Data/King/Perturbed_adatas/adata_Sham_1_center_345_1.h5ad
# python perturb.py --adata_path ../../../Data/King/Simulated_adatas/adata_Sham_1_center_345.h5ad --adata_save_path ../../../Data/King/Perturbed_adatas/adata_Sham_1_center_345_2.h5ad
# python perturb.py --adata_path ../../../Data/King/Simulated_adatas/adata_Sham_1_center_345.h5ad --adata_save_path ../../../Data/King/Perturbed_adatas/adata_Sham_1_center_345_3.h5ad

# python run_grid_search.py --dataset Simulation_v3 --adata_left_path ../../../Data/King/Fixed_adatas/adata_Sham_1.h5ad --adata_healthy_right_path 'None' --adata_right_path ../../../Data/King/Perturbed_adatas/adata_Sham_1_center_345_1.h5ad
# python run_grid_search.py --dataset Simulation_v3 --adata_left_path ../../../Data/King/Fixed_adatas/adata_Sham_1.h5ad --adata_healthy_right_path 'None' --adata_right_path ../../../Data/King/Perturbed_adatas/adata_Sham_1_center_345_2.h5ad
# python run_grid_search.py --dataset Simulation_v3 --adata_left_path ../../../Data/King/Fixed_adatas/adata_Sham_1.h5ad --adata_healthy_right_path 'None' --adata_right_path ../../../Data/King/Perturbed_adatas/adata_Sham_1_center_345_3.h5ad



# python perturb.py --adata_path ../../../Data/King/Simulated_adatas/adata_Sham_1_center_1024.h5ad --adata_save_path ../../../Data/King/Perturbed_adatas/adata_Sham_1_center_1024_1.h5ad
# python perturb.py --adata_path ../../../Data/King/Simulated_adatas/adata_Sham_1_center_1024.h5ad --adata_save_path ../../../Data/King/Perturbed_adatas/adata_Sham_1_center_1024_2.h5ad
# python perturb.py --adata_path ../../../Data/King/Simulated_adatas/adata_Sham_1_center_1024.h5ad --adata_save_path ../../../Data/King/Perturbed_adatas/adata_Sham_1_center_1024_3.h5ad

# python run_grid_search.py --dataset Simulation_v3 --adata_left_path ../../../Data/King/Fixed_adatas/adata_Sham_1.h5ad --adata_healthy_right_path 'None' --adata_right_path ../../../Data/King/Perturbed_adatas/adata_Sham_1_center_1024_1.h5ad
# python run_grid_search.py --dataset Simulation_v3 --adata_left_path ../../../Data/King/Fixed_adatas/adata_Sham_1.h5ad --adata_healthy_right_path 'None' --adata_right_path ../../../Data/King/Perturbed_adatas/adata_Sham_1_center_1024_2.h5ad
# python run_grid_search.py --dataset Simulation_v3 --adata_left_path ../../../Data/King/Fixed_adatas/adata_Sham_1.h5ad --adata_healthy_right_path 'None' --adata_right_path ../../../Data/King/Perturbed_adatas/adata_Sham_1_center_1024_3.h5ad


# python perturb.py --adata_path ../../../Data/Mouse_DMD/Fixed_adatas/mdx_adata.h5ad --adata_save_path ../../../Data/Mouse_DMD/Perturbed_adatas/mdx_adata_1.h5ad
# python perturb.py --adata_path ../../../Data/Mouse_DMD/Fixed_adatas/mdx_adata.h5ad --adata_save_path ../../../Data/Mouse_DMD/Perturbed_adatas/mdx_adata_2.h5ad
# python perturb.py --adata_path ../../../Data/Mouse_DMD/Fixed_adatas/mdx_adata.h5ad --adata_save_path ../../../Data/Mouse_DMD/Perturbed_adatas/mdx_adata_3.h5ad

# python run_grid_search.py --dataset Mouse_DMD_grid_search_v2 --adata_left_path ../../../Data/Mouse_DMD/Fixed_adatas/C57BL10_adata.h5ad --adata_healthy_right_path ../../../Data/Mouse_DMD/Fixed_adatas/DBA2J_adata.h5ad --adata_right_path ../../../Data/Mouse_DMD/Perturbed_adatas/mdx_adata_1.h5ad
# python run_grid_search.py --dataset Mouse_DMD_grid_search_v2 --adata_left_path ../../../Data/Mouse_DMD/Fixed_adatas/C57BL10_adata.h5ad --adata_healthy_right_path ../../../Data/Mouse_DMD/Fixed_adatas/DBA2J_adata.h5ad --adata_right_path ../../../Data/Mouse_DMD/Perturbed_adatas/mdx_adata_2.h5ad
# python run_grid_search.py --dataset Mouse_DMD_grid_search_v2 --adata_left_path ../../../Data/Mouse_DMD/Fixed_adatas/C57BL10_adata.h5ad --adata_healthy_right_path ../../../Data/Mouse_DMD/Fixed_adatas/DBA2J_adata.h5ad --adata_right_path ../../../Data/Mouse_DMD/Perturbed_adatas/mdx_adata_3.h5ad




python perturb.py --adata_path ../../../Data/Mouse_DMD/Fixed_adatas/D2mdx_adata.h5ad --adata_save_path ../../../Data/Mouse_DMD/Perturbed_adatas/D2mdx_adata_1.h5ad
python perturb.py --adata_path ../../../Data/Mouse_DMD/Fixed_adatas/D2mdx_adata.h5ad --adata_save_path ../../../Data/Mouse_DMD/Perturbed_adatas/D2mdx_adata_2.h5ad
python perturb.py --adata_path ../../../Data/Mouse_DMD/Fixed_adatas/D2mdx_adata.h5ad --adata_save_path ../../../Data/Mouse_DMD/Perturbed_adatas/D2mdx_adata_3.h5ad

python run_grid_search.py --dataset Mouse_DMD_grid_search_v3 --adata_left_path ../../../Data/Mouse_DMD/Fixed_adatas/DBA2J_adata.h5ad --adata_healthy_right_path ../../../Data/Mouse_DMD/Fixed_adatas/C57BL10_adata.h5ad --adata_right_path ../../../Data/Mouse_DMD/Perturbed_adatas/D2mdx_adata_1.h5ad
python run_grid_search.py --dataset Mouse_DMD_grid_search_v3 --adata_left_path ../../../Data/Mouse_DMD/Fixed_adatas/DBA2J_adata.h5ad --adata_healthy_right_path ../../../Data/Mouse_DMD/Fixed_adatas/C57BL10_adata.h5ad --adata_right_path ../../../Data/Mouse_DMD/Perturbed_adatas/D2mdx_adata_2.h5ad
python run_grid_search.py --dataset Mouse_DMD_grid_search_v3 --adata_left_path ../../../Data/Mouse_DMD/Fixed_adatas/DBA2J_adata.h5ad --adata_healthy_right_path ../../../Data/Mouse_DMD/Fixed_adatas/C57BL10_adata.h5ad --adata_right_path ../../../Data/Mouse_DMD/Perturbed_adatas/D2mdx_adata_3.h5ad





# python perturb.py --adata_path ../../../Data/Mouse_brain/h5ad/adata_mouse_brain_D3.h5ad --adata_save_path ../../../Data/Mouse_brain/Perturbed_adatas/adata_mouse_brain_D3_1.h5ad
# python perturb.py --adata_path ../../../Data/Mouse_brain/h5ad/adata_mouse_brain_D3.h5ad --adata_save_path ../../../Data/Mouse_brain/Perturbed_adatas/adata_mouse_brain_D3_2.h5ad
# python perturb.py --adata_path ../../../Data/Mouse_brain/h5ad/adata_mouse_brain_D3.h5ad --adata_save_path ../../../Data/Mouse_brain/Perturbed_adatas/adata_mouse_brain_D3_3.h5ad

# python run_grid_search.py --dataset Mouse_brain_grid_search --adata_left_path ../../../Data/Mouse_brain/h5ad/adata_mouse_brain_ctrl.h5ad --adata_healthy_right_path None --adata_right_path ../../../Data/Mouse_brain/Perturbed_adatas/adata_mouse_brain_D3_1.h5ad
# python run_grid_search.py --dataset Mouse_brain_grid_search --adata_left_path ../../../Data/Mouse_brain/h5ad/adata_mouse_brain_ctrl.h5ad --adata_healthy_right_path None --adata_right_path ../../../Data/Mouse_brain/Perturbed_adatas/adata_mouse_brain_D3_2.h5ad
# python run_grid_search.py --dataset Mouse_brain_grid_search --adata_left_path ../../../Data/Mouse_brain/h5ad/adata_mouse_brain_ctrl.h5ad --adata_healthy_right_path None --adata_right_path ../../../Data/Mouse_brain/Perturbed_adatas/adata_mouse_brain_D3_3.h5ad

