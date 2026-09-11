"""Statistical models over the filing record.

Three rules govern everything in this package, and they exist because a model
that breaks any one of them is worse than no model in a valuation context.

**Point in time.** Every feature is built through a knowledge date, so a model
trained or scored at a past date sees only what had been filed by then. The
primitive is already in ``edgar.CompanyFacts``; nothing here may bypass it.

**A baseline it must beat.** Every model reports the naive alternative beside
itself: last year's growth, the sector median, the base rate. A model that does
not beat its baseline is reported as not beating it, not quietly shipped.

**A model card.** Every fitted model carries what it was trained on, when, with
what features, and how it scored out of sample. An unauditable model has no
place beside a valuation whose every other number traces to a filing.
"""
