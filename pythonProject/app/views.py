from django.shortcuts import render, redirect
from .forms import ShipForm, RouteForm
from .models import Route

def route_form(request):
    if request.method == 'POST':
        form = RouteForm(request.POST)
        if form.is_valid():
            route = form.save()
            # Здесь вы можете вызвать функцию для расчета кратчайшего маршрута
            # и сохранить результат в поле shortest_distance модели Route
            return redirect('route_result', pk=route.pk)
    else:
        form = RouteForm()
    return render(request, 'app/route_form.html', {'form': form})

def route_result(request, pk):
    route = Route.objects.get(pk=pk)
    return render(request, 'app/route_result.html', {'route': route})