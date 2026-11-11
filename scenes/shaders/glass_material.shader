#extends "default_material.shader"

#block parameters
{{ super() }}
<InstanceParameters> grazing_fade: bool = false;
#endblock
