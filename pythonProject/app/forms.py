from django import forms
from .models import Ship, Route

class ShipForm(forms.ModelForm):
    class Meta:
        model = Ship
        fields = ['name', 'ice_class', 'draft']

class RouteForm(forms.ModelForm):
    ships = forms.ModelMultipleChoiceField(
        queryset=Ship.objects.all(),
        widget=forms.CheckboxSelectMultiple
    )

    class Meta:
        model = Route
        fields = ['start_port', 'end_port', 'ice_thickness', 'temperature', 'ships']