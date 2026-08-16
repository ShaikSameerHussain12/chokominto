from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    CUSTOMER_TYPES = [
        ('Residential', 'Residential'),
        ('Commercial', 'Commercial'),
        ('Industrial', 'Industrial'),
        ('Unknown', 'Unknown'),
    ]
    METER_TYPES = [
        ('Analog', 'Analog'),
        ('Digital', 'Digital'),
        ('Smart', 'Smart'),
        ('Unknown', 'Unknown'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    customer_id = models.CharField(max_length=50, unique=True, db_index=True)
    customer_type = models.CharField(max_length=50, choices=CUSTOMER_TYPES, default='Residential')
    location = models.CharField(max_length=150, db_index=True)
    meter_type = models.CharField(max_length=50, choices=METER_TYPES, default='Analog')
    phone = models.CharField(max_length=20, blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.customer_id} - {self.user.username} ({self.customer_type})"

class ConsumptionRecord(models.Model):
    PAYMENT_STATUSES = [
        ('Paid', 'Paid'),
        ('Unpaid', 'Unpaid'),
        ('Pending', 'Pending'),
    ]

    customer_id = models.CharField(max_length=50, db_index=True)
    date = models.DateField(db_index=True)
    consumption = models.FloatField()
    billing_amount = models.DecimalField(max_digits=10, decimal_places=2)
    payment_status = models.CharField(max_length=20, choices=PAYMENT_STATUSES, default='Pending')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('customer_id', 'date')
        ordering = ['-date']

    def __str__(self):
        return f"{self.customer_id} - {self.date}: {self.consumption} m³"

class DatasetUpload(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Processing', 'Processing'),
        ('Completed', 'Completed'),
        ('Failed', 'Failed'),
    ]

    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='data/uploads/')
    filename = models.CharField(max_length=255)
    row_count = models.IntegerField(default=0)
    upload_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.filename} ({self.upload_status}) - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"

class Prediction(models.Model):
    RISK_LEVELS = [
        ('LOW', 'Low Risk'),
        ('MEDIUM', 'Medium Risk'),
        ('HIGH', 'High Risk'),
    ]

    customer_id = models.CharField(max_length=50, db_index=True)
    model_name = models.CharField(max_length=50, db_index=True)
    predicted_class = models.IntegerField(choices=[(0, 'Normal'), (1, 'Suspicious')], db_index=True)
    probability = models.FloatField()
    risk_level = models.CharField(max_length=10, choices=RISK_LEVELS, db_index=True)
    key_indicators = models.TextField(help_text="JSON list of text reasons")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.customer_id} - {self.model_name}: {self.risk_level} ({self.probability:.2f})"

class Investigation(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Under Review', 'Under Review'),
        ('Confirmed Fraud', 'Confirmed Fraud'),
        ('False Positive', 'False Positive'),
        ('Meter Problem', 'Meter Problem'),
        ('Billing Problem', 'Billing Problem'),
        ('No Issue', 'No Issue'),
        ('Resolved', 'Resolved'),
    ]

    customer_id = models.CharField(max_length=50, db_index=True)
    status = models.CharField(max_length=30, choices=STATUS_CHOICES, default='Pending', db_index=True)
    remarks = models.TextField(blank=True, null=True)
    confirmed_fraud = models.BooleanField(default=False)
    investigated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    investigated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Investigation for {self.customer_id} - {self.status}"

class Feedback(models.Model):
    FEEDBACK_TYPES = [
        ('Consumption Issue', 'Consumption Issue'),
        ('Billing Issue', 'Billing Issue'),
        ('Meter Issue', 'Meter Issue'),
        ('False Fraud Alert', 'False Fraud Alert'),
        ('General Feedback', 'General Feedback'),
    ]
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Reviewed', 'Reviewed'),
        ('Resolved', 'Resolved'),
    ]

    customer_id = models.CharField(max_length=50, db_index=True)
    feedback_type = models.CharField(max_length=50, choices=FEEDBACK_TYPES)
    message = models.TextField()
    response = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Feedback ({self.feedback_type}) from {self.customer_id} - {self.status}"

class ModelRun(models.Model):
    model_name = models.CharField(max_length=50)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1 = models.FloatField()
    roc_auc = models.FloatField()
    training_time = models.FloatField()
    dataset_size = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.model_name} Run - F1: {self.f1:.2f} ({self.created_at.strftime('%Y-%m-%d')})"

class BlockedCustomer(models.Model):
    customer_id = models.CharField(max_length=50, unique=True, db_index=True)
    reason = models.TextField()
    blocked_by = models.ForeignKey(User, on_delete=models.CASCADE)
    blocked_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=True)

    def __str__(self):
        return f"Blocked {self.customer_id} - Active: {self.active}"
