import os
import csv
import json
import logging
import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import AuthenticationForm
from django.http import HttpResponse, Http404
from django.db.models import Q, Avg, StdDev, Max, Min, Count
from django.utils import timezone
from django.core.paginator import Paginator

from django.conf import settings
from core.models import (
    UserProfile, ConsumptionRecord, DatasetUpload, Prediction,
    Investigation, Feedback, ModelRun, BlockedCustomer
)
from core.forms import DatasetUploadForm, FeedbackForm, FeedbackResponseForm, InvestigationForm, UserProfileForm
from core import services

logger = logging.getLogger(__name__)

# Helpers for Role checks
def is_admin(user):
    return user.is_authenticated and (user.is_superuser or user.is_staff)

def is_customer(user):
    return user.is_authenticated and not (user.is_superuser or user.is_staff)

# Shared Views
def home_redirect(request):
    """Redirects user to respective dashboard based on credentials."""
    if not request.user.is_authenticated:
        return redirect('login')
    if is_admin(request.user):
        return redirect('dashboard')
    return redirect('user_profile')

def user_login(request):
    """User Login handler."""
    if request.user.is_authenticated:
        return redirect('home')
        
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {user.username}!")
                return redirect('home')
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
        
    return render(request, 'login.html', {'form': form})

@login_required
def user_logout(request):
    """User Logout handler."""
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('login')


# =========================================================================
# CUSTOMER VIEWS
# =========================================================================

@login_required
@user_passes_test(is_customer, login_url='dashboard')
def user_profile(request):
    """Customer Dashboard / Profile View."""
    profile = get_object_or_404(UserProfile, user=request.user)
    
    # Check if this customer is blocked
    is_blocked = BlockedCustomer.objects.filter(customer_id=profile.customer_id, active=True).exists()
    
    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Your profile has been updated.")
            return redirect('user_profile')
    else:
        form = UserProfileForm(instance=profile)
        
    # Get general consumption aggregate metrics for this customer
    records = ConsumptionRecord.objects.filter(customer_id=profile.customer_id)
    aggs = records.aggregate(
        avg=Avg('consumption'),
        max=Max('consumption'),
        min=Min('consumption'),
        count=Count('id')
    )
    
    # Get latest active prediction risk
    pred = Prediction.objects.filter(customer_id=profile.customer_id).first()
    
    context = {
        'profile': profile,
        'form': form,
        'aggs': aggs,
        'prediction': pred,
        'is_blocked': is_blocked
    }
    return render(request, 'user/profile.html', context)

@login_required
@user_passes_test(is_customer, login_url='dashboard')
def user_consumption(request):
    """Customer personal consumption timeline list."""
    profile = get_object_or_404(UserProfile, user=request.user)
    
    # Simple search & sort
    sort = request.GET.get('sort', '-date')
    records_qs = ConsumptionRecord.objects.filter(customer_id=profile.customer_id).order_by(sort)
    
    paginator = Paginator(records_qs, 12) # 1 year of monthly records per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'profile': profile,
        'page_obj': page_obj,
        'sort': sort
    }
    return render(request, 'user/consumption.html', context)

@login_required
@user_passes_test(is_customer, login_url='dashboard')
def user_graphs(request):
    """Customer personal consumption Chart.js page."""
    profile = get_object_or_404(UserProfile, user=request.user)
    # Order by date ascending for chronologically plotting lines
    records = ConsumptionRecord.objects.filter(customer_id=profile.customer_id).order_by('date')
    
    dates = [r.date.strftime('%Y-%m') for r in records]
    consumption = [float(r.consumption) for r in records]
    billing = [float(r.billing_amount) for r in records]
    
    context = {
        'profile': profile,
        'dates_json': json.dumps(dates),
        'consumption_json': json.dumps(consumption),
        'billing_json': json.dumps(billing)
    }
    return render(request, 'user/graphs.html', context)

@login_required
@user_passes_test(is_customer, login_url='dashboard')
def user_feedback(request):
    """Customer submits feedback & reviews old messages."""
    profile = get_object_or_404(UserProfile, user=request.user)
    feedbacks = Feedback.objects.filter(customer_id=profile.customer_id)
    
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            f = form.save(commit=False)
            f.customer_id = profile.customer_id
            f.save()
            messages.success(request, "Feedback submitted successfully. Administrators will review it shortly.")
            return redirect('user_feedback')
    else:
        form = FeedbackForm()
        
    context = {
        'profile': profile,
        'feedbacks': feedbacks,
        'form': form
    }
    return render(request, 'user/feedback.html', context)


# =========================================================================
# ADMIN VIEWS
# =========================================================================

@login_required
@user_passes_test(is_admin, login_url='login')
def admin_dashboard(request):
    """Admin operational dashboard."""
    # Check if we should seed demo data
    if request.GET.get('seed') == 'true':
        try:
            services.seed_demo_data()
            messages.success(request, "Database seeded successfully with demo customers and monthly consumption records!")
        except Exception as e:
            messages.error(request, f"Seeding failed: {str(e)}")
        return redirect('dashboard')
        
    # Standard KPIs
    total_customers = UserProfile.objects.count()
    total_records = ConsumptionRecord.objects.count()
    
    # Deduce stats from predictions
    latest_run = ModelRun.objects.first()
    best_model = latest_run.model_name if latest_run else 'None'
    
    # Latest predictions for each customer
    predictions_qs = Prediction.objects.values('customer_id').annotate(latest_id=Max('id'))
    latest_ids = [p['latest_id'] for p in predictions_qs]
    latest_preds = Prediction.objects.filter(id__in=latest_ids)
    
    analyzed_count = latest_preds.count()
    suspicious_count = latest_preds.filter(predicted_class=1).count()
    normal_count = latest_preds.filter(predicted_class=0).count()
    
    fraud_pct = (suspicious_count / analyzed_count * 100) if analyzed_count > 0 else 0.0
    
    # Top Suspicious customers
    top_suspicious = latest_preds.filter(predicted_class=1).order_by('-probability')[:5]
    
    # Enrich with profile metadata and latest investigations
    top_suspicious_list = []
    for p in top_suspicious:
        prof = UserProfile.objects.filter(customer_id=p.customer_id).first()
        inv = Investigation.objects.filter(customer_id=p.customer_id).first()
        top_suspicious_list.append({
            'prediction': p,
            'profile': prof,
            'investigation': inv
        })
        
    recent_investigations = Investigation.objects.order_by('-investigated_at')[:5]
    recent_feedbacks = Feedback.objects.order_by('-created_at')[:5]
    
    context = {
        'total_customers': total_customers,
        'total_records': total_records,
        'analyzed_count': analyzed_count,
        'suspicious_count': suspicious_count,
        'normal_count': normal_count,
        'fraud_pct': fraud_pct,
        'best_model': best_model,
        'top_suspicious': top_suspicious_list,
        'recent_investigations': recent_investigations,
        'recent_feedbacks': recent_feedbacks,
        'latest_run': latest_run
    }
    return render(request, 'dashboard/home.html', context)

@login_required
@user_passes_test(is_admin, login_url='login')
def dataset_upload(request):
    """Ingest consumption datasets CSV/Excel."""
    uploads = DatasetUpload.objects.all().order_by('-uploaded_at')
    
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save(commit=False)
            uploaded_file.uploaded_by = request.user
            uploaded_file.filename = request.FILES['file'].name
            uploaded_file.save()
            
            # Run ingestion task synchronously for this reconstruction
            try:
                success, row_count, updated_count = services.import_uploaded_dataset(uploaded_file.id)
                if success:
                    messages.success(request, f"File '{uploaded_file.filename}' processed successfully! Ingested {row_count} records (Updated: {updated_count}).")
                else:
                    messages.error(request, "Failed to process the dataset file.")
            except Exception as e:
                messages.error(request, f"Ingestion error: {str(e)}")
                
            return redirect('upload_dataset')
    else:
        form = DatasetUploadForm()
        
    return render(request, 'upload/upload.html', {'form': form, 'uploads': uploads})

@login_required
@user_passes_test(is_admin, login_url='login')
def model_training(request):
    """Triggers pipeline training and prints accuracy updates."""
    latest_run = ModelRun.objects.first()
    all_runs = ModelRun.objects.all()[:10]
    
    # Check if models/ folder contains trained artifacts
    models_dir = os.path.join(settings.BASE_DIR, 'models')
    models_exist = os.path.exists(os.path.join(models_dir, 'svm_model.joblib'))
    
    training_results = None
    
    if request.method == 'POST':
        test_size = float(request.POST.get('test_size', 0.20))
        random_state = int(request.POST.get('random_state', 42))
        
        try:
            # Execute full pipeline: Preprocess -> Feature Engineer -> Scale -> Train SVM & KNN -> Evaluate
            training_results = services.run_model_training_pipeline(test_size=test_size, random_state=random_state)
            messages.success(request, "Models successfully retrained and saved!")
            latest_run = ModelRun.objects.first()
        except Exception as e:
            messages.error(request, f"Training failed: {str(e)}")
            logger.exception("Model training error:")
            
    # Check if evaluation plot file exists, copy/link path
    confusion_matrix_url = None
    comparison_plot_url = None
    if os.path.exists(os.path.join(settings.BASE_DIR, 'media', 'charts', 'svm_confusion_matrix.png')):
        confusion_matrix_url = settings.MEDIA_URL + 'charts/svm_confusion_matrix.png'
    if os.path.exists(os.path.join(settings.BASE_DIR, 'media', 'charts', 'model_comparison.png')):
        comparison_plot_url = settings.MEDIA_URL + 'charts/model_comparison.png'
        
    context = {
        'latest_run': latest_run,
        'all_runs': all_runs,
        'models_exist': models_exist,
        'results': training_results,
        'confusion_matrix_url': confusion_matrix_url,
        'comparison_plot_url': comparison_plot_url
    }
    return render(request, 'training/training.html', context)

@login_required
@user_passes_test(is_admin, login_url='login')
def predictions_log(request):
    """Prediction triggers and tables log."""
    latest_predictions = Prediction.objects.order_by('-created_at')[:20]
    
    model_runs = ModelRun.objects.values('model_name').annotate(latest_date=Max('created_at'))
    models_trained = [mr['model_name'] for mr in model_runs]
    
    results = None
    
    if request.method == 'POST':
        model_name = request.POST.get('model_name', 'SVM')
        try:
            created, alerts = services.execute_fraud_predictions(model_name=model_name)
            messages.success(request, f"Prediction scoring completed! Processed {created} customers, flagged {alerts} new alerts.")
            latest_predictions = Prediction.objects.order_by('-created_at')[:20]
            
            # Fetch prediction totals
            predictions_qs = Prediction.objects.values('customer_id').annotate(latest_id=Max('id'))
            latest_ids = [p['latest_id'] for p in predictions_qs]
            latest_preds = Prediction.objects.filter(id__in=latest_ids)
            
            results = {
                'analyzed': latest_preds.count(),
                'normal': latest_preds.filter(predicted_class=0).count(),
                'suspicious': latest_preds.filter(predicted_class=1).count(),
                'high_risk': latest_preds.filter(risk_level='HIGH').count(),
                'med_risk': latest_preds.filter(risk_level='MEDIUM').count(),
                'low_risk': latest_preds.filter(risk_level='LOW').count(),
            }
        except Exception as e:
            messages.error(request, f"Prediction scoring failed: {str(e)}")
            logger.exception("Anomalies scoring error:")
            
    context = {
        'latest_predictions': latest_predictions,
        'models_trained': models_trained,
        'results': results
    }
    return render(request, 'predictions/predictions.html', context)

@login_required
@user_passes_test(is_admin, login_url='login')
def fraud_alerts(request):
    """Suspicious Customer Flags list view."""
    # Sub-query for latest predictions
    predictions_qs = Prediction.objects.values('customer_id').annotate(latest_id=Max('id'))
    latest_ids = [p['latest_id'] for p in predictions_qs]
    
    # Filter for suspicious predictions
    alerts_qs = Prediction.objects.filter(id__in=latest_ids, predicted_class=1)
    
    # Search and Filter
    q_search = request.GET.get('q', '')
    q_risk = request.GET.get('risk', '')
    q_status = request.GET.get('status', '')
    
    if q_search:
        alerts_qs = alerts_qs.filter(customer_id__icontains=q_search)
        
    if q_risk:
        alerts_qs = alerts_qs.filter(risk_level=q_risk)
        
    # Status filtering requires joining with Investigation
    alert_list = []
    for a in alerts_qs:
        prof = UserProfile.objects.filter(customer_id=a.customer_id).first()
        inv = Investigation.objects.filter(customer_id=a.customer_id).first()
        
        # Exclude/include based on status query
        if q_status:
            status_val = inv.status if inv else 'Pending'
            if status_val != q_status:
                continue
                
        alert_list.append({
            'prediction': a,
            'profile': prof,
            'investigation': inv
        })
        
    # Pagination
    paginator = Paginator(alert_list, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'q': q_search,
        'risk': q_risk,
        'status': q_status,
        'risk_choices': ['HIGH', 'MEDIUM', 'LOW'],
        'status_choices': [c[0] for c in Investigation.STATUS_CHOICES]
    }
    return render(request, 'predictions/fraud_alerts.html', context)

@login_required
@user_passes_test(is_admin, login_url='login')
def customer_list(request):
    """Admin searchable list of all customers."""
    profiles_qs = UserProfile.objects.all()
    
    # Filters
    q_search = request.GET.get('q', '')
    q_type = request.GET.get('type', '')
    q_loc = request.GET.get('location', '')
    
    if q_search:
        profiles_qs = profiles_qs.filter(
            Q(customer_id__icontains=q_search) | 
            Q(user__email__icontains=q_search)
        )
    if q_type:
        profiles_qs = profiles_qs.filter(customer_type=q_type)
    if q_loc:
        profiles_qs = profiles_qs.filter(location=q_loc)
        
    # Get locations and types for filters
    locations = UserProfile.objects.values_list('location', flat=True).distinct()
    types = [c[0] for c in UserProfile.CUSTOMER_TYPES]
    
    customer_data = []
    # Fetch latest predictions and blocked status
    for p in profiles_qs:
        pred = Prediction.objects.filter(customer_id=p.customer_id).first()
        is_blocked = BlockedCustomer.objects.filter(customer_id=p.customer_id, active=True).exists()
        
        avg_cons = ConsumptionRecord.objects.filter(customer_id=p.customer_id).aggregate(avg=Avg('consumption'))['avg'] or 0.0
        
        customer_data.append({
            'profile': p,
            'prediction': pred,
            'is_blocked': is_blocked,
            'avg_consumption': avg_cons
        })
        
    paginator = Paginator(customer_data, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'q': q_search,
        'type': q_type,
        'location_filter': q_loc,
        'locations': locations,
        'types': types
    }
    return render(request, 'customers/list.html', context)

@login_required
@user_passes_test(is_admin, login_url='login')
def customer_detail(request, customer_id):
    """Customer Detail page with consumption timeline, predictions, and investigation updates."""
    profile = get_object_or_404(UserProfile, customer_id=customer_id)
    records = ConsumptionRecord.objects.filter(customer_id=customer_id).order_by('date')
    
    # Aggregates
    aggs = records.aggregate(
        avg=Avg('consumption'),
        max=Max('consumption'),
        min=Min('consumption'),
        std=StdDev('consumption')
    )
    
    # Model predictions
    prediction = Prediction.objects.filter(customer_id=customer_id).first()
    indicators = []
    if prediction and prediction.key_indicators:
        try:
            indicators = json.loads(prediction.key_indicators)
        except Exception:
            indicators = [prediction.key_indicators]
            
    # Investigation status
    investigation, inv_created = Investigation.objects.get_or_create(
        customer_id=customer_id,
        defaults={'status': 'Pending'}
    )
    
    # Block status
    is_blocked = BlockedCustomer.objects.filter(customer_id=customer_id, active=True).exists()
    
    # Form processing
    if request.method == 'POST':
        form = InvestigationForm(request.POST, instance=investigation)
        if form.is_valid():
            inv = form.save(commit=False)
            inv.investigated_by = request.user
            inv.save()
            messages.success(request, f"Investigation notes updated for customer {customer_id}.")
            return redirect('customer_detail', customer_id=customer_id)
    else:
        form = InvestigationForm(instance=investigation)
        
    feedbacks = Feedback.objects.filter(customer_id=customer_id)
    
    # Plotting arrays
    dates = [r.date.strftime('%Y-%m') for r in records]
    consumption = [float(r.consumption) for r in records]
    billing = [float(r.billing_amount) for r in records]
    
    context = {
        'profile': profile,
        'records': records,
        'aggs': aggs,
        'prediction': prediction,
        'indicators': indicators,
        'investigation': investigation,
        'is_blocked': is_blocked,
        'form': form,
        'feedbacks': feedbacks,
        'dates_json': json.dumps(dates),
        'consumption_json': json.dumps(consumption),
        'billing_json': json.dumps(billing)
    }
    return render(request, 'customers/detail.html', context)

@login_required
@user_passes_test(is_admin, login_url='login')
def toggle_block_customer(request, customer_id):
    """Blocks or unblocks a customer account."""
    profile = get_object_or_404(UserProfile, customer_id=customer_id)
    blocked_entry = BlockedCustomer.objects.filter(customer_id=customer_id, active=True).first()
    
    if blocked_entry:
        blocked_entry.active = False
        blocked_entry.save()
        messages.success(request, f"Customer account {customer_id} has been unblocked.")
    else:
        reason = request.POST.get('reason', 'Suspicious activity flagged by model and confirmed by administrator.')
        BlockedCustomer.objects.create(
            customer_id=customer_id,
            reason=reason,
            blocked_by=request.user,
            active=True
        )
        messages.warning(request, f"Customer account {customer_id} has been blocked.")
        
    return redirect('customer_detail', customer_id=customer_id)

@login_required
@user_passes_test(is_admin, login_url='login')
def admin_feedback_list(request):
    """List of all submitted user feedbacks."""
    feedbacks_qs = Feedback.objects.all()
    
    q_status = request.GET.get('status', '')
    if q_status:
        feedbacks_qs = feedbacks_qs.filter(status=q_status)
        
    paginator = Paginator(feedbacks_qs, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'status_filter': q_status,
        'status_choices': ['Pending', 'Reviewed', 'Resolved']
    }
    return render(request, 'feedback/list.html', context)

@login_required
@user_passes_test(is_admin, login_url='login')
def admin_feedback_detail(request, feedback_id):
    """View details of a customer feedback and respond."""
    feedback = get_object_or_404(Feedback, id=feedback_id)
    profile = UserProfile.objects.filter(customer_id=feedback.customer_id).first()
    
    if request.method == 'POST':
        form = FeedbackResponseForm(request.POST, instance=feedback)
        if form.is_valid():
            f = form.save(commit=False)
            f.save()
            messages.success(request, f"Response sent to customer {feedback.customer_id}.")
            return redirect('admin_feedback')
    else:
        form = FeedbackResponseForm(instance=feedback)
        
    return render(request, 'feedback/detail.html', {'feedback': feedback, 'profile': profile, 'form': form})

@login_required
@user_passes_test(is_admin, login_url='login')
def investigations_list(request):
    """Track inspections pipeline."""
    investigations_qs = Investigation.objects.all().order_by('-investigated_at')
    
    q_status = request.GET.get('status', '')
    if q_status:
        investigations_qs = investigations_qs.filter(status=q_status)
        
    paginator = Paginator(investigations_qs, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'status_filter': q_status,
        'status_choices': [c[0] for c in Investigation.STATUS_CHOICES]
    }
    return render(request, 'investigations/list.html', context)

@login_required
@user_passes_test(is_admin, login_url='login')
def graph_analysis(request):
    """Dynamic ChartJS plotting variables."""
    # 1. Class distribution (Fraud vs Normal)
    predictions_qs = Prediction.objects.values('customer_id').annotate(latest_id=Max('id'))
    latest_ids = [p['latest_id'] for p in predictions_qs]
    latest_preds = Prediction.objects.filter(id__in=latest_ids)
    
    suspicious = latest_preds.filter(predicted_class=1).count()
    normal = latest_preds.filter(predicted_class=0).count()
    
    # 2. Risk breakdown
    risk_high = latest_preds.filter(risk_level='HIGH').count()
    risk_med = latest_preds.filter(risk_level='MEDIUM').count()
    risk_low = latest_preds.filter(risk_level='LOW').count()
    
    # 3. Monthly consumption trend
    # Group by month and calculate average consumption
    # We will slice dates into YYYY-MM
    db_records = ConsumptionRecord.objects.all()
    # To be fast and database engine agnostic, we aggregate using python for the graph page
    # (since DateField grouping varies between sqlite and mysql)
    monthly_data = {}
    for r in db_records:
        month_str = r.date.strftime('%Y-%m')
        if month_str not in monthly_data:
            monthly_data[month_str] = []
        monthly_data[month_str].append(r.consumption)
        
    sorted_months = sorted(monthly_data.keys())
    avg_consumption_by_month = [float(np.mean(monthly_data[m])) for m in sorted_months]
    
    # 4. Location-wise alert count
    location_alerts = {}
    for a in latest_preds.filter(predicted_class=1):
        prof = UserProfile.objects.filter(customer_id=a.customer_id).first()
        loc = prof.location if prof else 'Unknown'
        location_alerts[loc] = location_alerts.get(loc, 0) + 1
        
    # 5. SVM vs KNN comparison
    runs = ModelRun.objects.all()
    svm_run = runs.filter(model_name='SVM').first()
    knn_run = runs.filter(model_name='KNN').first()
    
    svm_metrics = [
        svm_run.accuracy * 100 if svm_run else 0,
        svm_run.precision * 100 if svm_run else 0,
        svm_run.recall * 100 if svm_run else 0,
        svm_run.f1 * 100 if svm_run else 0
    ]
    knn_metrics = [
        knn_run.accuracy * 100 if knn_run else 0,
        knn_run.precision * 100 if knn_run else 0,
        knn_run.recall * 100 if knn_run else 0,
        knn_run.f1 * 100 if knn_run else 0
    ]
    
    context = {
        'suspicious': suspicious,
        'normal': normal,
        'risk_high': risk_high,
        'risk_med': risk_med,
        'risk_low': risk_low,
        'months_json': json.dumps(sorted_months),
        'monthly_avg_json': json.dumps(avg_consumption_by_month),
        'location_labels_json': json.dumps(list(location_alerts.keys())),
        'location_values_json': json.dumps(list(location_alerts.values())),
        'svm_metrics_json': json.dumps(svm_metrics),
        'knn_metrics_json': json.dumps(knn_metrics)
    }
    return render(request, 'graph_analysis/graphs.html', context)

@login_required
@user_passes_test(is_admin, login_url='login')
def admin_reports(request):
    """Summary and CSV exporting."""
    # Model Run metrics
    latest_run = ModelRun.objects.first()
    
    # Stats
    total_customers = UserProfile.objects.count()
    
    predictions_qs = Prediction.objects.values('customer_id').annotate(latest_id=Max('id'))
    latest_ids = [p['latest_id'] for p in predictions_qs]
    latest_preds = Prediction.objects.filter(id__in=latest_ids)
    
    suspicious_count = latest_preds.filter(predicted_class=1).count()
    confirmed_fraud = Investigation.objects.filter(status='Confirmed Fraud').count()
    false_positives = Investigation.objects.filter(status='False Positive').count()
    
    # Export CSV action
    if request.GET.get('export') == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="waterguard_fraud_predictions_{timezone.now().date()}.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['Customer ID', 'Model', 'Predicted Class', 'Anomaly Probability', 'Risk Level', 'Key Indicators', 'Scored At'])
        
        for p in latest_preds:
            writer.writerow([
                p.customer_id,
                p.model_name,
                'Suspicious' if p.predicted_class == 1 else 'Normal',
                p.probability,
                p.risk_level,
                p.key_indicators,
                p.created_at.strftime('%Y-%m-%d %H:%M:%S')
            ])
        return response
        
    context = {
        'latest_run': latest_run,
        'total_customers': total_customers,
        'suspicious_count': suspicious_count,
        'confirmed_fraud': confirmed_fraud,
        'false_positives': false_positives,
    }
    return render(request, 'reports/reports.html', context)
