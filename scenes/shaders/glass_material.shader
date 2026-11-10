#extends "default_material.shader"

#block parameters
{{ super() }}
{{ parameter("grazing_fade", "bool", false, "instance") }}
#endblock
