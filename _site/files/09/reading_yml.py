import yaml

filename = "serialized_yaml_1.yml"

my_dict = yaml.load(open(filename))

print(my_dict)
