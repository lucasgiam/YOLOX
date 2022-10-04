from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()   # client_secrets.json need to be in the same directory as the script
drive = GoogleDrive(gauth)

# View all folders and file in your Google Drive
fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
for file in fileList:
    print('Title: %s, ID: %s' % (file['title'], file['id']))
    # Get the folder ID that you want
    if(file['title'] == "sp_ppe_1_videos"):
        fileID = file['id']

# Test uploading a video file to Google Drive folder
file1 = drive.CreateFile({'title':'test_vid.mp4', 'parents': [{'kind': 'drive#fileLink', 'id': fileID}]})
file1.SetContentFile(r'C:\Users\Lucas_Giam\Pictures\Camera Roll\WIN_20221004_16_39_49_Pro.mp4')   # enter path to actual image file here
file1.Upload()
print('Created file %s with mimeType %s' % (file1['title'], file1['mimeType']))

# Generate URL to file
link1 = file1['alternateLink']
print(link1)