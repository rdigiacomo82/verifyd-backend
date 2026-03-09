import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield, 
  AlertCircle, 
  CheckCircle2, 
  FileCheck, 
  Calendar, 
  Upload as UploadIcon, 
  Link as LinkIcon, 
  X, 
  Loader2,
  FileVideo,
  Download,
  MailCheck,
  ExternalLink,
  Copy
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useToast } from '@/components/ui/use-toast';
import { useEmailVerificationFlow } from '@/hooks/useEmailVerificationFlow';
import LimitReachedOverlay from '@/components/LimitReachedOverlay';
import EmailVerificationFlow from '@/components/EmailVerificationFlow';
import Logo from '@/components/Logo';

const MAX_FILE_SIZE = 52428800; // 50MB

const loadingMessages = [
  "Analyzing your video...",
  "Analyzing signal patterns...",
  "Running AI vision check...",
  "Calculating authenticity score..."
];

const AppPage = () => {
  const { toast } = useToast();
  
  // Shared State
  const [email, setEmail] = useState('');
  const [isEmailValid, setIsEmailValid] = useState(false);
  const [emailTouched, setEmailTouched] = useState(false);
  const [pendingAction, setPendingAction] = useState(null); // 'upload' | 'analyze' | null
  const [copied, setCopied] = useState(false);
  const [messageIndex, setMessageIndex] = useState(0);

  // Email Verification Flow State
  const { 
    isVerifying,
    showModal: showVerifyModal,
    error: verifyError,
    isVerified,
    sendOTP,
    verifyOTP,
    isEmailVerified,
    closeVerificationModal
  } = useEmailVerificationFlow();

  // Link Analysis State
  const [videoLink, setVideoLink] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [linkResult, setLinkResult] = useState(null);
  const [linkError, setLinkError] = useState(null);

  // File Upload State
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [processingStatus, setProcessingStatus] = useState('idle'); // idle, processing, error
  const [uploadResult, setUploadResult] = useState(null);
  
  // Polling State
  const [jobState, setJobState] = useState(null);
  const [queuePosition, setQueuePosition] = useState(null);
  
  const fileInputRef = useRef(null);
  const emailInputRef = useRef(null);

  // Limits State
  const [showLimitOverlay, setShowLimitOverlay] = useState(false);

  // Check verification on email change
  useEffect(() => {
    if (isEmailValid && email) {
      isEmailVerified(email);
    }
  }, [email, isEmailValid, isEmailVerified]);

  // Rotating Messages Effect
  useEffect(() => {
    let interval;
    if (processingStatus === 'processing') {
      interval = setInterval(() => {
        setMessageIndex((prev) => (prev + 1) % loadingMessages.length);
      }, 8000);
    } else {
      setMessageIndex(0);
    }
    return () => clearInterval(interval);
  }, [processingStatus]);

  // Email Validation Logic
  const validateEmail = (email) => {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(String(email).toLowerCase());
  };

  const handleEmailChange = (e) => {
    const val = e.target.value;
    setEmail(val);
    const isValid = validateEmail(val);
    setIsEmailValid(isValid);
    if (!emailTouched && val.length > 0) setEmailTouched(true);
  };

  const handleDownload = (downloadUrl) => {
    window.location.href = downloadUrl;
  };

  const handleCopyLink = async (url) => {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast({
        title: "Copied!",
        description: "Certificate URL copied to clipboard.",
        className: "bg-green-600 text-white border-none",
        duration: 2000,
      });
    } catch (err) {
      toast({
        variant: "destructive",
        title: "Copy Failed",
        description: "Could not copy the URL. Please try manually.",
      });
    }
  };

  const handleVerifyEmailClick = async () => {
    if (!isEmailValid) return;
    emailInputRef.current?.blur(); // Dismiss keyboard
    const result = await sendOTP(email);
    if (result.success && !result.alreadyVerified) {
      toast({
        title: "Code Sent",
        description: `Verification code sent to ${email}`,
        className: "bg-amber-500 text-black border-none",
      });
    } else if (result.success && result.alreadyVerified) {
      toast({
        title: "Already Verified",
        description: "Your email is already verified!",
        className: "bg-green-600 text-white border-none",
      });
    } else {
      toast({
        variant: "destructive",
        title: "Verification Error",
        description: result.error || "Please try again later.",
      });
    }
  };

  const handleVerificationSuccess = () => {
    emailInputRef.current?.blur(); // Dismiss keyboard
    toast({
      title: "Email Verified",
      description: "Your email has been successfully verified.",
      className: "bg-green-600 text-white border-none",
    });
    
    // Resume pending action
    if (pendingAction === 'upload') {
      performUpload();
    } else if (pendingAction === 'analyze') {
      performAnalyze();
    }
    setPendingAction(null);
  };

  const checkVerification = async (action) => {
    if (!isEmailValid) {
      toast({
          variant: "destructive",
          title: "Email Required",
          description: "Please enter a valid email address first.",
      });
      return false;
    }
    
    if (!isEmailVerified(email)) {
      setPendingAction(action);
      await handleVerifyEmailClick();
      return false;
    }
    return true;
  };

  // --- LINK ANALYSIS LOGIC ---
  const pollLinkJobStatus = async (jobId, attempts = 0) => {
    if (attempts >= 40) {
      setLinkError("Analysis took too long. Please try again.");
      setIsAnalyzing(false);
      toast({
        variant: "destructive",
        title: "Timeout",
        description: "Analysis took too long. Please try again.",
      });
      return;
    }

    try {
      const response = await fetch(`https://verifyd-backend.onrender.com/job-status/${jobId}`);
      
      if (response.status === 404) {
        setTimeout(() => pollLinkJobStatus(jobId, attempts + 1), 3000);
        return;
      }
      
      if (!response.ok) throw new Error('Failed to fetch job status');
      
      const data = await response.json();
      
      if (data.status === 'complete') {
        setLinkResult(data.result || data);
        setIsAnalyzing(false);
        toast({
          title: "Analysis Complete",
          description: "Link analyzed successfully.",
          className: "bg-green-500 border-none text-white",
        });
      } else if (data.status === 'error') {
        throw new Error(data.error || 'Job processing failed');
      } else {
        setTimeout(() => pollLinkJobStatus(jobId, attempts + 1), 3000);
      }
    } catch (error) {
      setLinkError(error.message || "There was an error processing your link.");
      setIsAnalyzing(false);
      toast({
        variant: "destructive",
        title: "Analysis Failed",
        description: error.message || "There was an error processing your link.",
      });
    }
  };

  const performAnalyze = async () => {
    setIsAnalyzing(true);
    setLinkResult(null);
    setLinkError(null);

    try {
      const url = `https://verifyd-backend.onrender.com/analyze-link/?video_url=${encodeURIComponent(videoLink)}&email=${encodeURIComponent(email)}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        if (response.status === 402 || response.status === 403) {
          const errData = await response.json().catch(() => ({}));
          if (errData.error === "limit_reached") throw new Error("LIMIT_REACHED");
          if (errData.error === "email_not_verified") throw new Error("EMAIL_NOT_VERIFIED");
        }
        throw new Error("Failed to analyze link");
      }

      const data = await response.json();

      if (data.status === 'queued' && data.job_id) {
        pollLinkJobStatus(data.job_id);
      } else if (data.job_id) {
        pollLinkJobStatus(data.job_id);
      } else {
        setLinkResult(data.result || data);
        setIsAnalyzing(false);
        toast({
          title: "Analysis Complete",
          description: "Link analyzed successfully.",
          className: "bg-green-500 border-none text-white",
        });
      }
    } catch (error) {
      if (error.message === "LIMIT_REACHED") {
        setIsAnalyzing(false);
        setShowLimitOverlay(true);
        return;
      }
      if (error.message === "EMAIL_NOT_VERIFIED") {
        setIsAnalyzing(false);
        setPendingAction('analyze');
        handleVerifyEmailClick();
        return;
      }
      
      setLinkError(error.message || "There was an error analyzing your link.");
      setIsAnalyzing(false);
      toast({
        variant: "destructive",
        title: "Analysis Failed",
        description: error.message || "There was an error analyzing your link.",
      });
    }
  };

  const handleLinkAnalyze = async () => {
    if (!videoLink) {
      toast({
        variant: "destructive",
        title: "Link Required",
        description: "Please paste a video link to analyze.",
      });
      return;
    }
    if (await checkVerification('analyze')) {
      performAnalyze();
    }
  };

  const clearLinkState = () => {
    setLinkResult(null);
    setLinkError(null);
    setVideoLink('');
    setIsAnalyzing(false);
  };

  // --- FILE UPLOAD LOGIC ---
  const processSelectedFile = async (selectedFile) => {
    if (selectedFile.size > MAX_FILE_SIZE) {
      toast({
        variant: "destructive",
        title: "File exceeds 50MB limit",
        description: "Your video is larger than our 50MB limit. Try compressing it or splitting it into smaller segments.",
        duration: 6000,
      });
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    setFile(selectedFile);
    setProcessingStatus('idle');
    setUploadResult(null);
    setJobState(null);
    setQueuePosition(null);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isEmailValid) return;

    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isEmailValid) return;
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (['video/mp4', 'video/quicktime', 'video/webm'].includes(droppedFile.type)) {
        await processSelectedFile(droppedFile);
      } else {
        toast({
          variant: "destructive",
          title: "Invalid file type",
          description: "Please upload MP4, MOV, or WebM files.",
        });
      }
    }
  };

  const handleFileSelect = async (e) => {
    if (e.target.files && e.target.files[0]) {
      await processSelectedFile(e.target.files[0]);
    }
  };

  const pollJobStatus = async (jobId, attempts = 0) => {
    if (attempts >= 40) {
      setProcessingStatus('error');
      setJobState(null);
      toast({
        variant: "destructive",
        title: "Timeout",
        description: "Analysis took too long. Please try again.",
      });
      return;
    }

    try {
      const response = await fetch(`https://verifyd-backend.onrender.com/job-status/${jobId}`);
      
      if (response.status === 404) {
        setTimeout(() => pollJobStatus(jobId, attempts + 1), 3000);
        return;
      }
      
      if (!response.ok) throw new Error('Failed to fetch job status');
      
      const data = await response.json();
      
      if (data.status === 'complete') {
        setUploadResult(data.result || data);
        setProcessingStatus('idle');
        setJobState(null);
        toast({
          title: "Analysis Complete",
          description: "Video processed successfully.",
          className: "bg-green-500 border-none text-white",
        });
      } else if (data.status === 'error') {
        throw new Error(data.error || 'Job processing failed');
      } else if (data.status === 'not_found') {
        setTimeout(() => pollJobStatus(jobId, attempts + 1), 3000);
      } else {
        setJobState(data.status);
        if (data.status === 'queued' && data.position) {
          setQueuePosition(data.position);
        } else {
          setQueuePosition(null);
        }
        setTimeout(() => pollJobStatus(jobId, attempts + 1), 3000);
      }
    } catch (error) {
      setProcessingStatus('error');
      setJobState(null);
      toast({
        variant: "destructive",
        title: "Analysis Failed",
        description: error.message || "There was an error processing your video.",
      });
    }
  };

  const performUpload = async () => {
    if (!file) return;

    setProcessingStatus('processing');
    setUploadResult(null);
    setJobState('analyzing');
    setQueuePosition(null);
    
    const formData = new FormData();
    formData.append('email', email);
    formData.append('file', file);

    try {
      const data = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', 'https://verifyd-backend.onrender.com/upload/');

        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const responseData = JSON.parse(xhr.responseText);
              resolve(responseData);
            } catch (err) {
              reject(new Error("Invalid response format"));
            }
          } else {
            if (xhr.status === 402 || xhr.status === 403) {
              try {
                const errData = JSON.parse(xhr.responseText);
                if (errData.error === "limit_reached") {
                  reject(new Error("LIMIT_REACHED"));
                  return;
                }
                if (errData.error === "email_not_verified") {
                  reject(new Error("EMAIL_NOT_VERIFIED"));
                  return;
                }
              } catch (parseErr) {}
            }
            reject(new Error(`Upload failed: ${xhr.statusText || xhr.status}`));
          }
        };

        xhr.onerror = () => {
          reject(new Error("Network error occurred during upload"));
        };

        xhr.send(formData);
      });

      if (data.status === 'queued' && data.job_id) {
        setJobState('queued');
        pollJobStatus(data.job_id);
      } else if (data.job_id) {
        pollJobStatus(data.job_id);
      } else {
        setUploadResult(data.result || data);
        setProcessingStatus('idle');
        setJobState(null);
        toast({
          title: "Analysis Complete",
          description: "Video processed successfully.",
          className: "bg-green-500 border-none text-white",
        });
      }

    } catch (error) {
      if (error.message === "LIMIT_REACHED") {
        setProcessingStatus('idle');
        setJobState(null);
        setShowLimitOverlay(true);
        return;
      }
      if (error.message === "EMAIL_NOT_VERIFIED") {
        setProcessingStatus('idle');
        setJobState(null);
        setPendingAction('upload');
        handleVerifyEmailClick();
        return;
      }
      
      toast({
        variant: "destructive",
        title: "Upload Failed",
        description: error.message || "There was an error uploading your video.",
      });
      setProcessingStatus('error');
      setUploadResult({ error: error.message });
      setJobState(null);
    }
  };

  const uploadVideo = async (e) => {
    e.preventDefault();
    if (await checkVerification('upload')) {
      performUpload();
    }
  };

  const clearFile = () => {
    setFile(null);
    setProcessingStatus('idle');
    setUploadResult(null);
    setJobState(null);
    setQueuePosition(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const getStatusColor = (status, score) => {
    if (!status && score === undefined) return 'text-white';
    const s = status ? status.toLowerCase() : '';
    if (s.includes('authentic') || (score !== undefined && score >= 85)) return 'text-[#10B981]';
    if (s.includes('suspicious') || (score !== undefined && score >= 50 && score < 85)) return 'text-[#FBBF24]';
    if (s.includes('fake') || (score !== undefined && score < 50)) return 'text-[#EF4444]';
    return 'text-white';
  };

  const handleTouchStart = (e) => {
    if (
      document.activeElement === emailInputRef.current && 
      e.target !== emailInputRef.current
    ) {
      document.activeElement?.blur();
    }
  };

  const showResultArea = file || uploadResult || processingStatus !== 'idle';

  return (
    <div 
      className="min-h-screen bg-[#0C0D0D] text-white pt-16 pb-4 md:pt-24 md:pb-12 px-2 sm:px-4 md:px-6 lg:px-8 relative flex flex-col justify-center"
      onTouchStart={handleTouchStart}
    >
      
      {/* Verify Modal */}
      {showVerifyModal && (
        <EmailVerificationFlow 
          email={email}
          onClose={closeVerificationModal}
          onVerificationSuccess={handleVerificationSuccess}
          verifyOTP={verifyOTP}
          sendOTP={sendOTP}
          isVerifying={isVerifying}
          error={verifyError}
        />
      )}

      {/* Limit Reached Overlay */}
      <AnimatePresence>
        {showLimitOverlay && (
          <LimitReachedOverlay onDismiss={() => setShowLimitOverlay(false)} />
        )}
      </AnimatePresence>

      {/* Full-Screen Gold Loading Overlay */}
      <AnimatePresence>
        {processingStatus === 'processing' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-black/80 backdrop-blur-md"
          >
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
              className="w-28 h-28 rounded-full border-4 border-white/10 border-t-[#D4AF37] border-r-[#D4AF37] mb-8 shadow-[0_0_30px_rgba(212,175,55,0.3)]"
            />
            <div className="h-16 relative flex justify-center items-center w-full max-w-xs px-6">
              <AnimatePresence mode="wait">
                <motion.p
                  key={messageIndex}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.5 }}
                  className="text-xl font-semibold text-[#D4AF37] absolute text-center w-full drop-shadow-md"
                >
                  {loadingMessages[messageIndex]}
                </motion.p>
              </AnimatePresence>
            </div>
            <p className="text-gray-400 text-sm mt-6 px-8 text-center">
              Videos can take up to a minute — we're being thorough! ✨
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="max-w-6xl mx-auto space-y-4 md:space-y-12 w-full">
        {/* Header Section */}
        <div className="text-center space-y-2 md:space-y-6">
            <div className="flex justify-center mb-1 md:mb-6">
              <Logo className="scale-90 sm:scale-100 md:scale-110" />
            </div>
            
            <p className="text-gray-400 text-xs sm:text-sm md:text-lg max-w-2xl mx-auto px-4">
              Enter your email below to unlock the verification tools.
            </p>

            {/* Email Input */}
            <div className="max-w-md mx-auto relative pt-1 md:pt-4">
              <div className={`relative transition-all duration-300`}>
                <div className={`absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg blur opacity-20 ${isEmailValid ? 'opacity-40' : 'opacity-20'}`}></div>
                <div className="relative bg-[#1A1D1E] rounded-lg border border-white/10 flex items-center shadow-2xl p-1">
                  <div className="pl-3 md:pl-4 text-gray-400">
                    <Shield className="w-4 h-4 md:w-5 md:h-5" />
                  </div>
                  <Input
                    ref={emailInputRef}
                    type="email"
                    placeholder="Enter your email"
                    value={email}
                    onChange={handleEmailChange}
                    className="border-0 bg-transparent focus-visible:ring-0 text-sm md:text-lg py-2 md:py-6 px-2 md:px-4 text-white placeholder:text-gray-500 w-full h-10 md:h-auto"
                  />
                  <div className="pr-1 md:pr-2 flex items-center gap-1 md:gap-2">
                    {isEmailValid && !isVerified && (
                      <Button 
                        size="sm" 
                        onClick={handleVerifyEmailClick}
                        disabled={isVerifying}
                        className="bg-amber-500 hover:bg-amber-600 text-black font-semibold h-7 md:h-9 px-2 md:px-3 text-xs md:text-sm rounded-md shadow-lg shadow-amber-500/20 whitespace-nowrap"
                      >
                        {isVerifying ? <Loader2 className="w-3 h-3 md:w-4 md:h-4 animate-spin" /> : "Verify"}
                      </Button>
                    )}
                    {isVerified ? (
                      <div className="flex items-center justify-center h-7 md:h-9 px-2 bg-green-500/10 rounded-md border border-green-500/30">
                        <MailCheck className="w-4 h-4 md:w-5 md:h-5 text-green-500" />
                        <span className="ml-1 text-[10px] md:text-xs text-green-500 font-medium hidden sm:inline-block">Verified</span>
                      </div>
                    ) : (
                       isEmailValid ? <CheckCircle2 className="w-4 h-4 md:w-5 md:h-5 text-blue-400" /> : (emailTouched && email.length > 0 && <AlertCircle className="w-4 h-4 md:w-5 md:h-5 text-red-500" />)
                    )}
                  </div>
                </div>
              </div>
              {!isEmailValid && emailTouched && email.length > 0 && (
                <p className="text-red-400 text-[10px] md:text-sm mt-1 md:mt-2 font-medium">
                  Please enter a valid email
                </p>
              )}
            </div>
        </div>

        {/* Main Content Columns */}
        <div className={`grid grid-cols-1 lg:grid-cols-2 gap-3 md:gap-8 items-start transition-opacity duration-300 ${!isEmailValid ? 'opacity-50 pointer-events-none' : 'opacity-100'}`}>
          
          {/* Left Column: Upload & Results */}
          <div className="relative group bg-[#1A1D1E] rounded-xl border-2 border-amber-500/40 p-4 md:p-8 transition-all duration-300 shadow-xl overflow-hidden min-h-[180px] md:min-h-[400px]">
            <div className="flex items-center space-x-2 md:space-x-3 mb-2 md:mb-6">
              <div className="p-1.5 md:p-2 bg-blue-500/10 rounded-lg">
                <UploadIcon className="w-4 h-4 md:w-6 md:h-6 text-blue-400" />
              </div>
              <h2 className="text-base md:text-xl font-semibold text-white">Upload Video</h2>
            </div>

            <AnimatePresence mode="wait">
              {!showResultArea ? (
                <motion.div 
                  key="dropzone"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className={`border-2 border-dashed rounded-xl p-3 md:p-8 text-center transition-all h-[120px] md:h-[300px] flex flex-col justify-center items-center ${dragActive ? 'border-blue-500 bg-blue-500/5' : 'border-white/10 hover:border-white/20'}`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  <input 
                    ref={fileInputRef}
                    type="file" 
                    className="hidden" 
                    accept="video/mp4,video/quicktime,video/webm"
                    onChange={handleFileSelect}
                    disabled={!isEmailValid}
                  />
                  <div className="w-8 h-8 md:w-16 md:h-16 bg-gray-800 rounded-full flex items-center justify-center mb-1 md:mb-4">
                    <FileVideo className="w-4 h-4 md:w-8 md:h-8 text-gray-400" />
                  </div>
                  <p className="text-white text-xs md:text-base font-medium mb-0.5 md:mb-2">Drag & Drop Video</p>
                  <p className="text-[10px] md:text-sm text-gray-500 mb-2 md:mb-6 hidden sm:block">MP4, MOV, WebM (Max 50MB)</p>
                  
                  <Button 
                    variant="outline" 
                    onClick={() => {
                      if (!isVerified) {
                        setPendingAction(null);
                        handleVerifyEmailClick();
                      } else {
                        fileInputRef.current?.click();
                      }
                    }}
                    disabled={!isEmailValid}
                    className="bg-white text-black hover:bg-gray-200 border-none px-4 md:px-8 font-bold text-xs md:text-sm h-8 md:h-11 shadow-lg transition-all duration-200 flex items-center justify-center"
                  >
                    Select File
                  </Button>
                </motion.div>
              ) : (
                <motion.div 
                    key="result"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="bg-black/30 rounded-xl p-3 md:p-6 border border-white/10 flex flex-col justify-between h-auto min-h-[160px] md:min-h-[300px]"
                >
                  {!uploadResult && processingStatus !== 'error' && (
                    <div className="flex justify-between items-start mb-2 md:mb-6">
                      <div className="flex items-center space-x-2 md:space-x-3 overflow-hidden">
                          <div className="p-1.5 md:p-2 bg-blue-500/20 rounded-lg">
                          <FileVideo className="w-4 h-4 md:w-5 md:h-5 text-blue-400" />
                          </div>
                          <div className="truncate">
                          <p className="text-white text-xs md:text-sm font-medium truncate max-w-[100px] md:max-w-[150px]">{file?.name}</p>
                          <p className="text-[10px] md:text-xs text-gray-500">{file ? (file.size / (1024*1024)).toFixed(2) : '0'} MB</p>
                          </div>
                      </div>
                      
                      {processingStatus !== 'processing' && (
                        <button type="button" onClick={clearFile} className="text-gray-500 hover:text-white transition-colors">
                          <X className="w-4 h-4 md:w-5 md:h-5" />
                        </button>
                      )}
                    </div>
                  )}

                  {processingStatus === 'idle' && !uploadResult && (
                    <div className="mt-auto">
                      <Button 
                        onClick={uploadVideo}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-4 md:py-6 text-sm md:text-base h-10 md:h-auto"
                        disabled={!isEmailValid}
                      >
                        Start Certification Process
                      </Button>
                      <p className="text-[10px] md:text-xs text-center text-gray-500 mt-2 md:mt-3">This will upload your video to our secure server.</p>
                    </div>
                  )}

                  {processingStatus === 'processing' && (
                    <div className="flex flex-col items-center justify-center space-y-2 md:space-y-6 my-2 md:my-8 py-2 md:py-4 w-full text-center relative overflow-hidden">
                      <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-full h-full bg-amber-500/5 blur-[30px] md:blur-[50px] pointer-events-none"></div>
                      <div className="relative w-10 h-10 md:w-16 md:h-16 flex items-center justify-center z-10">
                        <div className="absolute inset-0 rounded-full border-[3px] md:border-4 border-[#222]"></div>
                        <div className="absolute inset-0 rounded-full border-[3px] md:border-4 border-[#f59e0b] border-t-transparent animate-[spin_1s_linear_infinite]"></div>
                      </div>
                      
                      <div className="z-10 px-2 mt-2 md:mt-6 space-y-1 md:space-y-2">
                        <h3 className="text-sm md:text-[18px] font-bold text-white leading-tight">
                          Analyzing...
                        </h3>
                        <p className="text-[10px] md:text-[15px] text-[#f59e0b]">Videos can take up to a minute — we're being thorough! ✨</p>
                        
                        {jobState === 'queued' && queuePosition && (
                          <p className="text-[10px] md:text-sm text-amber-400 font-bold animate-pulse mt-1">Position {queuePosition} in queue</p>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {processingStatus === 'idle' && uploadResult && (
                     <motion.div 
                       initial={{ opacity: 0, y: 10 }}
                       animate={{ opacity: 1, y: 0 }}
                       transition={{ duration: 0.3 }}
                       className="flex flex-col items-center justify-center space-y-3 md:space-y-6 w-full"
                     >
                       {uploadResult.video_url && (
                          <div className="w-full bg-[#0C0D0D] rounded-lg overflow-hidden border border-white/20 md:border-2 md:border-white shadow-lg">
                            <video 
                              controls
                              playsInline
                              preload="metadata"
                              className="w-full max-h-[150px] md:max-h-[250px] object-contain"
                            >
                              <source src={uploadResult.video_url} type="video/mp4" />
                            </video>
                          </div>
                       )}

                       <div className="relative flex flex-col items-center mt-1 md:mt-2">
                         <div className={`w-16 h-16 md:w-24 md:h-24 rounded-full flex items-center justify-center border-[3px] md:border-4 ${getStatusColor(uploadResult.status, uploadResult.authenticity_score).replace('text-', 'border-')}`}>
                           <span className={`text-xl md:text-3xl font-bold ${getStatusColor(uploadResult.status, uploadResult.authenticity_score)}`}>
                             {uploadResult.authenticity_score || 0}%
                           </span>
                         </div>
                         <div className="mt-2 md:mt-4 text-center">
                           <h3 className={`text-base md:text-2xl font-bold uppercase tracking-wide ${getStatusColor(uploadResult.status, uploadResult.authenticity_score)}`}>
                             {uploadResult.status || "Analyzed"}
                           </h3>
                         </div>
                       </div>

                       <div className="flex flex-col gap-2 w-full mt-2">
                         {uploadResult.certificate_id && (
                           <>
                             <Button 
                               onClick={() => handleDownload(`https://verifyd-backend.onrender.com/download/${uploadResult.certificate_id}`)}
                               className="w-full bg-[#10B981] hover:bg-[#059669] text-white gap-2 font-bold py-4 md:py-6 text-sm md:text-md shadow-lg h-10 md:h-auto"
                             >
                               <Download className="w-4 h-4 md:w-5 md:h-5" />
                               Download Video
                             </Button>

                             {/* Share link — prominent input + copy button */}
                             <div className="flex flex-col items-start mt-3 w-full bg-black/40 p-3 md:p-4 rounded-lg border border-purple-500/40">
                               <p className="font-bold text-sm md:text-base text-white mb-2 w-full">
                                 🔗 Share your certified video link
                               </p>
                               <div className="flex items-center gap-2 w-full">
                                 <input
                                   readOnly
                                   type="text"
                                   value={`https://vfvid.com/v/${uploadResult.certificate_id}`}
                                   onClick={(e) => e.target.select()}
                                   style={{
                                     flex: 1,
                                     background: '#111827',
                                     border: '1px solid #7c3aed',
                                     borderRadius: '8px',
                                     color: '#c4b5fd',
                                     padding: '8px 10px',
                                     fontSize: '13px',
                                     outline: 'none',
                                     cursor: 'text',
                                     minWidth: 0,
                                   }}
                                 />
                                 <button
                                   onClick={() => handleCopyLink(`https://vfvid.com/v/${uploadResult.certificate_id}`)}
                                   style={{
                                     background: copied ? '#059669' : '#7c3aed',
                                     color: 'white',
                                     border: 'none',
                                     borderRadius: '8px',
                                     padding: '8px 14px',
                                     cursor: 'pointer',
                                     fontWeight: 'bold',
                                     fontSize: '13px',
                                     whiteSpace: 'nowrap',
                                     flexShrink: 0,
                                     transition: 'background 0.2s',
                                   }}
                                 >
                                   {copied ? '✓ Copied!' : '📋 Copy'}
                                 </button>
                               </div>
                             </div>
                           </>
                         )}
                         
                         {/* Scan another video — bright and visible */}
                         <button
                           onClick={clearFile}
                           style={{
                             width: '100%',
                             marginTop: '8px',
                             padding: '14px',
                             background: 'transparent',
                             border: '2px solid rgba(255,255,255,0.5)',
                             borderRadius: '10px',
                             color: 'white',
                             fontSize: '15px',
                             fontWeight: 'bold',
                             cursor: 'pointer',
                             display: 'flex',
                             alignItems: 'center',
                             justifyContent: 'center',
                             gap: '8px',
                             transition: 'border-color 0.2s, background 0.2s',
                           }}
                           onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.08)'}
                           onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                         >
                           ↺ Scan Another Video
                         </button>
                       </div>
                     </motion.div>
                  )}

                  {processingStatus === 'error' && (
                    <motion.div 
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="flex flex-col items-center justify-center space-y-2 md:space-y-4 my-2 md:my-8 p-4 md:p-6 bg-[#1A1D1E] rounded-xl border border-red-500/20"
                    >
                      <AlertCircle className="w-8 h-8 md:w-12 md:h-12 text-[#EF4444]" />
                      <div className="text-center space-y-1 md:space-y-2">
                        <h3 className="text-base md:text-xl font-bold text-white">Analysis Failed</h3>
                        <p className="text-gray-400 text-xs md:text-sm max-w-sm">
                          {uploadResult?.error || "Error processing video. Try again."}
                        </p>
                      </div>
                      <button
                        onClick={clearFile}
                        style={{
                          width: '100%',
                          marginTop: '8px',
                          padding: '14px',
                          background: 'transparent',
                          border: '2px solid rgba(255,255,255,0.5)',
                          borderRadius: '10px',
                          color: 'white',
                          fontSize: '15px',
                          fontWeight: 'bold',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: '8px',
                        }}
                      >
                        ↺ Try Again
                      </button>
                    </motion.div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Right Column: Link Analysis */}
          <div className="relative bg-[#1A1D1E] rounded-xl border-2 border-amber-500/40 p-4 md:p-8 transition-all duration-300 shadow-xl flex flex-col min-h-[140px] md:min-h-[465px]">
            <div className="flex items-center space-x-2 md:space-x-3 mb-3 md:mb-6">
              <div className="p-1.5 md:p-2 bg-purple-500/10 rounded-lg">
                <LinkIcon className="w-4 h-4 md:w-6 md:h-6 text-purple-400" />
              </div>
              <h2 className="text-base md:text-xl font-semibold text-white">Paste a Link</h2>
            </div>

            <div className="space-y-3 md:space-y-4 flex-1 flex flex-col">
              {isAnalyzing ? (
                <div className="flex-1 flex flex-col items-center justify-center space-y-6">
                  <div className="relative w-24 h-24 flex items-center justify-center z-10">
                    <div className="absolute inset-0 rounded-full border-[3px] md:border-4 border-[#222]"></div>
                    <div className="absolute inset-0 rounded-full border-[3px] md:border-4 border-[#f59e0b] border-t-transparent animate-[spin_1s_linear_infinite]"></div>
                  </div>
                  <div className="text-center">
                    <h3 className="text-lg font-bold text-white mb-1">Analyzing...</h3>
                    <p className="text-sm text-[#f59e0b]">This may take a minute</p>
                  </div>
                </div>
              ) : linkResult ? (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex flex-col items-center justify-center flex-1 space-y-3 md:space-y-6"
                >
                  <div className="relative flex flex-col items-center mt-1 md:mt-2">
                    <div className={`w-16 h-16 md:w-24 md:h-24 rounded-full flex items-center justify-center border-[3px] md:border-4 ${getStatusColor(linkResult.status, linkResult.authenticity_score).replace('text-', 'border-')}`}>
                      <span className={`text-xl md:text-3xl font-bold ${getStatusColor(linkResult.status, linkResult.authenticity_score)}`}>
                        {linkResult.authenticity_score || 0}%
                      </span>
                    </div>
                    <div className="mt-2 md:mt-4 text-center">
                      <h3 className={`text-base md:text-2xl font-bold uppercase tracking-wide ${getStatusColor(linkResult.status, linkResult.authenticity_score)}`}>
                        {linkResult.status || "Analyzed"}
                      </h3>
                    </div>
                  </div>
                  <div className="flex flex-col gap-2 w-full mt-2">
                    {linkResult.certificate_id && (
                      <>
                        <Button 
                          onClick={() => handleDownload(`https://verifyd-backend.onrender.com/download/${linkResult.certificate_id}`)}
                          className="w-full bg-[#10B981] hover:bg-[#059669] text-white gap-2 font-bold py-4 md:py-6 text-sm md:text-md shadow-lg h-10 md:h-auto"
                        >
                          <Download className="w-4 h-4 md:w-5 md:h-5" />
                          Download Video
                        </Button>

                        {/* Share link — prominent input + copy button */}
                        <div className="flex flex-col items-start mt-3 w-full bg-black/40 p-3 md:p-4 rounded-lg border border-purple-500/40">
                          <p className="font-bold text-sm md:text-base text-white mb-2 w-full">
                            🔗 Share your certified video link
                          </p>
                          <div className="flex items-center gap-2 w-full">
                            <input
                              readOnly
                              type="text"
                              value={`https://vfvid.com/v/${linkResult.certificate_id}`}
                              onClick={(e) => e.target.select()}
                              style={{
                                flex: 1,
                                background: '#111827',
                                border: '1px solid #7c3aed',
                                borderRadius: '8px',
                                color: '#c4b5fd',
                                padding: '8px 10px',
                                fontSize: '13px',
                                outline: 'none',
                                cursor: 'text',
                                minWidth: 0,
                              }}
                            />
                            <button
                              onClick={() => handleCopyLink(`https://vfvid.com/v/${linkResult.certificate_id}`)}
                              style={{
                                background: copied ? '#059669' : '#7c3aed',
                                color: 'white',
                                border: 'none',
                                borderRadius: '8px',
                                padding: '8px 14px',
                                cursor: 'pointer',
                                fontWeight: 'bold',
                                fontSize: '13px',
                                whiteSpace: 'nowrap',
                                flexShrink: 0,
                                transition: 'background 0.2s',
                              }}
                            >
                              {copied ? '✓ Copied!' : '📋 Copy'}
                            </button>
                          </div>
                        </div>
                      </>
                    )}
                    <button
                      onClick={clearLinkState}
                      style={{
                        width: '100%',
                        marginTop: '8px',
                        padding: '14px',
                        background: 'transparent',
                        border: '2px solid rgba(255,255,255,0.5)',
                        borderRadius: '10px',
                        color: 'white',
                        fontSize: '15px',
                        fontWeight: 'bold',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '8px',
                        transition: 'border-color 0.2s, background 0.2s',
                      }}
                      onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.08)'}
                      onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                    >
                      ↺ Scan Another Link
                    </button>
                  </div>
                </motion.div>
              ) : linkError ? (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="flex flex-col items-center justify-center space-y-2 md:space-y-4 flex-1 p-4 md:p-6 bg-[#1A1D1E] rounded-xl border border-red-500/20"
                >
                  <AlertCircle className="w-8 h-8 md:w-12 md:h-12 text-[#EF4444]" />
                  <div className="text-center space-y-1 md:space-y-2">
                    <h3 className="text-base md:text-xl font-bold text-white">Analysis Failed</h3>
                    <p className="text-gray-400 text-xs md:text-sm max-w-sm">
                      {linkError}
                    </p>
                  </div>
                  <button
                    onClick={clearLinkState}
                    style={{
                      width: '100%',
                      marginTop: '8px',
                      padding: '14px',
                      background: 'transparent',
                      border: '2px solid rgba(255,255,255,0.5)',
                      borderRadius: '10px',
                      color: 'white',
                      fontSize: '15px',
                      fontWeight: 'bold',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '8px',
                    }}
                  >
                    ↺ Try Again
                  </button>
                </motion.div>
              ) : (
                <>
                  <Input
                    placeholder="https://tiktok.com/..."
                    value={videoLink}
                    onChange={(e) => setVideoLink(e.target.value)}
                    disabled={!isEmailValid}
                    className="bg-black/20 border-white/10 text-white focus:border-purple-500/50 h-10 md:h-auto text-sm md:text-base"
                  />
                  
                  <Button 
                    onClick={handleLinkAnalyze}
                    disabled={!isEmailValid || !videoLink || isAnalyzing}
                    className="w-full bg-white text-black hover:bg-gray-200 font-bold h-10 md:h-12 text-sm md:text-md disabled:opacity-70 flex items-center justify-center"
                  >
                      Analyze Link
                  </Button>

                  <div className="mt-2 pt-2 md:mt-8 md:pt-8 border-t border-white/10 flex-1 flex flex-col justify-center items-center text-center">
                      <div className="text-gray-500 text-[10px] md:text-sm">
                        <p className="hidden sm:block">Supported platforms:</p>
                        <div className="flex justify-center space-x-2 md:space-x-3 mt-1 md:mt-2 opacity-60 text-[9px] md:text-sm">
                          <span>TikTok</span>•<span>Instagram</span>•<span>YouTube</span>
                        </div>
                      </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Footer Info */}
        <div className="flex flex-wrap md:grid md:grid-cols-3 justify-center gap-2 md:gap-6 pt-3 md:pt-8 border-t border-white/5 pb-2">
          <div className="flex items-center justify-center space-x-1.5 md:space-x-4 text-gray-400">
            <Calendar className="w-3 h-3 md:w-5 md:h-5 text-gray-500" />
            <span className="text-[10px] md:text-sm">Real-time Analysis</span>
          </div>
           <div className="flex items-center justify-center space-x-1.5 md:space-x-4 text-gray-400 hidden sm:flex">
            <FileCheck className="w-3 h-3 md:w-5 md:h-5 text-gray-500" />
            <span className="text-[10px] md:text-sm">Detailed Report</span>
          </div>
           <div className="flex items-center justify-center space-x-1.5 md:space-x-4 text-gray-400">
            <Shield className="w-3 h-3 md:w-5 md:h-5 text-gray-500" />
            <span className="text-[10px] md:text-sm">100% Secure</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AppPage;
