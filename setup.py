from setuptools import  find_packages, setup

def get_requirements(filename):
    with open(filename, 'r') as file:
        file_content = file.readlines()
    file_content = [data.replace("\n", "") for data in file_content]
    if '-e .' in file_content: file_content.remove('-e .')
    # print("Content: ", file_content)

setup(
    name            = "air_pollution_index_prediction",
    version         = "0.0.1",
    author          = "Tarun Kumar",
    author_email    = "tarun94060sharma@gmail.com",
    packages        = find_packages(),
    install_reuires = get_requirements("requirements.txt")
)