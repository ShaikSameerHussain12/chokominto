from django.contrib import admin
from core.models import (
    UserProfile, ConsumptionRecord, DatasetUpload, Prediction,
    Investigation, Feedback, ModelRun, BlockedCustomer
)

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('customer_id', 'user', 'customer_type', 'location', 'meter_type', 'created_at')
    search_fields = ('customer_id', 'user__username', 'user__email', 'location')
    list_filter = ('customer_type', 'meter_type', 'location')

@admin.register(ConsumptionRecord)
class ConsumptionRecordAdmin(admin.ModelAdmin):
    list_display = ('customer_id', 'date', 'consumption', 'billing_amount', 'payment_status')
    search_fields = ('customer_id', 'date')
    list_filter = ('payment_status', 'date')
    ordering = ('-date',)

@admin.register(DatasetUpload)
class DatasetUploadAdmin(admin.ModelAdmin):
    list_display = ('filename', 'uploaded_by', 'row_count', 'upload_status', 'uploaded_at')
    list_filter = ('upload_status', 'uploaded_at')
    search_fields = ('filename', 'uploaded_by__username')

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('customer_id', 'model_name', 'predicted_class', 'probability', 'risk_level', 'created_at')
    list_filter = ('model_name', 'predicted_class', 'risk_level', 'created_at')
    search_fields = ('customer_id',)
    ordering = ('-created_at',)

@admin.register(Investigation)
class InvestigationAdmin(admin.ModelAdmin):
    list_display = ('customer_id', 'status', 'confirmed_fraud', 'investigated_by', 'investigated_at')
    list_filter = ('status', 'confirmed_fraud', 'investigated_at')
    search_fields = ('customer_id', 'remarks')

@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('customer_id', 'feedback_type', 'status', 'created_at')
    list_filter = ('feedback_type', 'status', 'created_at')
    search_fields = ('customer_id', 'message', 'response')

@admin.register(ModelRun)
class ModelRunAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'dataset_size', 'created_at')
    list_filter = ('model_name', 'created_at')
    ordering = ('-created_at',)

@admin.register(BlockedCustomer)
class BlockedCustomerAdmin(admin.ModelAdmin):
    list_display = ('customer_id', 'blocked_by', 'blocked_at', 'active')
    list_filter = ('active', 'blocked_at')
    search_fields = ('customer_id', 'reason')
