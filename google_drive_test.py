# View all folders and file in your Google Drive
fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
for file in fileList:
    print('Title: %s, ID: %s' % (file['title'], file['id']))
    # Get the folder ID that you want
    if(file['title'] == "To Share"):
        fileID = file['id']

file1 = drive.CreateFile({"mimeType": "text/csv", "parents": [{"kind": "drive#fileLink", "id": fileID}]})
file1.SetContentFile("small_file.csv")
file1.Upload() # Upload the file.
print('Created file %s with mimeType %s' % (file1['title'], file1['mimeType'])) 