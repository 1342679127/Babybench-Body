# Babybench-Body
Body ownership experience experiment in babybench environment
Babybench is a virtual environment developed under the MIMo environment to simulate infant development. For details, please visit the website
https://babybench.github.io/2025/api/#file-structure



Install as follows
Create a conda environment
conda create --name babybench python=3.12
conda activate babybench

Clone the BabyBench2025 repository
git clone https://github.com/babybench/BabyBench2025_Starter_Kit.git
cd BabyBench2025_Starter_Kit

Install requirements
pip install -r requirements.txt

Install MIMo
pip install -e MIMo

All done! You are ready to start using BabyBench.

Launch the installation test
python test_installation.py





Recommended configurations
Besides the aforementioned config_test_installation.yml file, we also provide a configuration files for each of the two behaviors. These are not mandatory to use, but rather recommendations that we think provide a good starting point for training MIMo.

SELF-TOUCH
The configuration file for self-touch is config_selftouch.yml. It uses the crib scene. All the proprioception settings are set to True, whereas the vestibular and vision settings are set to False. The touch observations are enabled everywhere in MIMo’s body except for his eyes. The actuation_model is set to spring_damper, and the actuators of MIMo’s hands, fingers, arms, legs, body, and head are enabled, but not his feet or eyes, which are locked.

A number of simplifications are available here. In particular, you can reduce the size of the action space by disabling some of the actuators, e.g. the body or legs. You can also reduce the size of the observation space by disabling some of the parameters of the proprioception module, e.g. torques and limits. Finally, you can simplify the touch observations by using a higher scale factor, which will reduce the number of touch sensors, e.g. setting touch_scale to 5.





















