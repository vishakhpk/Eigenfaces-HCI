from __future__ import unicode_literals

from django.db import models

class Person(models.Model):
	person_id = models.IntegerField(primary_key=True)
	person_name = models.CharField(max_length=30)
	person_details = models.CharField(max_length=50, default="-")
