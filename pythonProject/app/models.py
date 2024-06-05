from django.db import models

class Ship(models.Model):
    name = models.CharField(max_length=100)
    ice_class = models.CharField(max_length=50)
    draft = models.FloatField()

class Route(models.Model):
    start_port = models.CharField(max_length=100)
    end_port = models.CharField(max_length=100)
    ice_thickness = models.FloatField()
    temperature = models.FloatField()
    ships = models.ManyToManyField(Ship)
    shortest_distance = models.FloatField(null=True, blank=True)