from django.contrib import admin

from .models import SkillAssessment


class SkillAssessmentAdmin(admin.ModelAdmin):
    list_display = (
        "assessment_name",
        "participant_name",
        "test_name",
        "is_registered",
        "created_at",
    )
    search_fields = (
        "assessment_name",
        "participant_name",
        "test_name",
    )
    list_filter = (
        "is_registered",
        "created_at",
    )


admin.site.register(SkillAssessment, SkillAssessmentAdmin)
