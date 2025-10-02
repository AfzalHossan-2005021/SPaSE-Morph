# SPaSE

1. If anaconda or miniconda is not installed then first install it otherwise skip step 1.

   For Linux:

   ```
   mkdir -p ~/miniconda3
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
   bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   rm ~/miniconda3/miniconda.sh
   ```

   After installing, close and reopen your terminal application or refresh it by running the following command:

   ```
   source ~/miniconda3/bin/activate
   ```

   Then, initialize conda on all available shells by running the following command:

   ```
   conda init --all
   ```

   See [Miniconda Installation Guide](https://www.anaconda.com/docs/getting-started/miniconda/install) for detailed instructions.

2. Clone the repo
   ```
   git clone https://github.com/AfzalHossan-2005021/SPaSE-Morph.git
   ```
3. Change directory to SPaSE-Morph
   ```
   cd SPaSE-Morph
   ```
4. Install the spase_test conda environment:
   ```
   conda env create -f environment.yml
   ```
5. Activate the conda environment:
   ```
   conda activate spase
   ```
6. Change directory to src
   ```
   cd src
   ```
7. Run grid_search
   ```
   python run_grid_search.py -d Mouse_Liver -l healthy -r pod -hr healthy_r
   ```
