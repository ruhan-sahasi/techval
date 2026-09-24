"""The investing dashboard: a private ledger, quotes, and one rendered page.

Everything in this package is a pure function of the owner's portfolio file,
the price series and the committed model panels, so a run is reproducible and
the tests never touch the network. Nothing here is advice: every valuation
figure ships beside its baseline and the verdict the model earned.
"""
