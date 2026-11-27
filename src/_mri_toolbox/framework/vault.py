from collections import UserDict

class Vault(UserDict):
    def __str__(self):
        return "Vault content is hidden from print()"
