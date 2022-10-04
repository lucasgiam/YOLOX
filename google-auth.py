from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LoadCredentialsFile("mycreds.txt")   # try to load saved client credentials
if gauth.credentials is None:
    gauth.LocalWebserverAuth()             # need to authenticate if mycreds.txt is not there, also ensure that client_secrets.json is in the same directory as this script
elif gauth.access_token_expired:
    gauth.Refresh()                        # refresh credentials if expired
else:
    gauth.Authorize()                      # initialize the saved credentials
gauth.SaveCredentialsFile("mycreds.txt")   # save the current credentials to a file
drive = GoogleDrive(gauth)

# View all folders and file in your Google Drive
fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
for file in fileList:
    print('Title: %s, ID: %s' % (file['title'], file['id']))
    # Get the folder ID that you want
    if(file['title'] == "sp_ppe_1_videos"):
        fileID = file['id']

# Test uploading a video file to Google Drive folder
file1 = drive.CreateFile({'title':'testing.mp4', 'parents': [{'kind': 'drive#fileLink', 'id': fileID}]})
file1.SetContentFile(r'C:\Users\Admin\Pictures\Camera Roll\testing.mp4')   # enter path to actual image file here
file1.Upload()
print('Created file %s with mimeType %s' % (file1['title'], file1['mimeType']))

# Generate URL to file
link1 = file1['alternateLink']
print(link1)