from django.contrib import admin
from .models import employes

# Register your models here.

class employesAdmin(admin.ModelAdmin):
    fieldsets = [
        (None, {'fields': ['employe_photo']}),
        (None, {'fields': ['employe_name']}),
        (None, {'fields': ['employe_id']}),
    ]
    # inlines = [ChoiceInline]
    # list_filter = ['pub_date']
    # search_fields = ['question_text']

    list_display = ('employe_photo', 'employe_name', 'employe_id')

admin.site.register(employes, employesAdmin)
