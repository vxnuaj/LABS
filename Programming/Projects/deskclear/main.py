'''

Run the script, it clears all items from your desktop that include screenshots, files, etc,

Moves them to either:

- trash
- downloads
- permanently deletes

'''

import os

def clear(filetypes, outcome):
     dir = os.chdir('/Users/juanvera/desktop')
     print(dir)

clear('.png', 'trash')


