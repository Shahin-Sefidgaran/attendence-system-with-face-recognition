# Generated by Django 3.0.5 on 2020-06-20 08:30

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='employes',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('employe_photo', models.FileField(upload_to='')),
                ('employe_name', models.CharField(max_length=255)),
                ('employe_id', models.CharField(max_length=10)),
            ],
        ),
    ]
