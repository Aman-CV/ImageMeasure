from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

print(default_storage.__class__)
default_storage.save("django5.txt", ContentFile("fixed"))

url = default_storage.url("django5.txt")
print(url)