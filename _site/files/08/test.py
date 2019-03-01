import re

text0 = """
<note>
  <to>Tove</to>
  <from>Jani</from>
  <heading>Reminder</heading>
  <body>Don't forget me this weekend!</body>
</note>
"""

### EXAMPLE 1 - find/replace
##
##text = re.sub("<[^<]+>", "", text0)
##
##print(text)


# EXAMPLE 2 - split

results = re.split("</[^<]+>", text0)

##for r in results:
##    r = re.sub("^\s+<[^<]+>|\n", "", r)
##    print(r)
##    input()
##
##print(results)


# EXAMPLE 3 - capture

text = re.search(r"(<from>)([^<]+)(</from>)", text0).group(1)

print(text)
