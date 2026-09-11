"""Technology, Media and Telecom sector layer.

The generic engine values any US filer. This package holds what is true of TMT
specifically: the sub-vertical taxonomy a coverage banker actually uses, the
operating metrics the sector is priced on and which do not appear in standard
XBRL tags, segment economics for the conglomerates, and the precedent
transactions that anchor an M&A discussion.

The separation is deliberate. Revenue is revenue everywhere; annual recurring
revenue, net revenue retention, average revenue per user and content
amortisation are not, and a module that mixed the two would quietly apply a
software convention to a tower REIT.
"""
