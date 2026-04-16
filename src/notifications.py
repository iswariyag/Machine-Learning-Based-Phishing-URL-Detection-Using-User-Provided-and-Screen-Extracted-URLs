# src/notifications.py
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGraphicsDropShadowEffect, QFrame
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PyQt6.QtGui import QColor, QFont

class RoundIcon(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self.setText(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFont(QFont("Segoe UI Emoji", 20))
        self.setStyleSheet("""
            QLabel {
                background-color: #333333;
                color: white;
                border-radius: 20px;
            }
        """)

class DynamicIsland(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Window Flags
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.Tool |
            Qt.WindowType.ToolTip 
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # Increased Height for Explanation
        self.width = 400
        self.height = 110 
        self.resize(self.width, self.height)

        self.setStyleSheet("""
            QWidget#Container {
                background-color: rgba(245, 245, 245, 0.98);
                border-radius: 18px;
                border: 1px solid rgba(0, 0, 0, 0.1);
            }
            QLabel { color: #333; border: none; font-family: 'Segoe UI', sans-serif; }
            QLabel#Title { font-weight: bold; font-size: 14px; }
            QLabel#Message { font-size: 13px; color: #555; }
            QLabel#ReasonHeader { font-weight: bold; font-size: 11px; color: #FF453A; }
            QLabel#ReasonText { font-size: 11px; color: #666; font-style: italic; }
        """)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.container = QFrame()
        self.container.setObjectName("Container")
        container_layout = QHBoxLayout(self.container)
        container_layout.setContentsMargins(15, 12, 15, 12)
        container_layout.setSpacing(15)

        # Icon
        self.icon_label = RoundIcon("🛡️")
        container_layout.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignTop)

        # Text Layout
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        
        self.title_label = QLabel("ShieldAI Protection")
        self.title_label.setObjectName("Title")
        
        self.message_label = QLabel("Threat Detected")
        self.message_label.setObjectName("Message")
        
        # --- NEW: Explanation Section ---
        self.reason_header = QLabel("SUSPICIOUS BECAUSE:")
        self.reason_header.setObjectName("ReasonHeader")
        
        self.reason_label = QLabel("Analyzing...")
        self.reason_label.setObjectName("ReasonText")
        self.reason_label.setWordWrap(True)

        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.message_label)
        text_layout.addSpacing(4)
        text_layout.addWidget(self.reason_header)
        text_layout.addWidget(self.reason_label)
        
        container_layout.addLayout(text_layout)
        main_layout.addWidget(self.container)

        # Shadow & Animation
        shadow = QGraphicsDropShadowEffect(self.container)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 5)
        self.container.setGraphicsEffect(shadow)

        self.animation = QPropertyAnimation(self, b"pos")
        self.animation.setDuration(500)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide_notification)

    def show_message(self, title, message, explanation="", color="#ff4d4d"):
        screen = self.screen().availableGeometry()
        x_pos = screen.width() - self.width - 20
        y_start = -self.height - 20
        y_end = 40 

        self.title_label.setText(title)
        self.title_label.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {color};")
        
        # Truncate main message if too long
        if len(message) > 40: message = message[:37] + "..."
        self.message_label.setText(message)
        
        # Set Explanation
        if explanation:
            self.reason_header.show()
            self.reason_label.show()
            self.reason_label.setText(explanation)
        else:
            self.reason_header.hide()
            self.reason_label.hide()

        self.show()
        self.raise_()
        self.activateWindow()

        self.animation.setStartValue(QPoint(x_pos, y_start))
        self.animation.setEndValue(QPoint(x_pos, y_end))
        self.animation.start()
        
        self.timer.start(8000) # Show for 8 seconds

    def hide_notification(self):
        screen = self.screen().availableGeometry()
        x_pos = screen.width() - self.width - 20
        self.animation.setStartValue(self.pos())
        self.animation.setEndValue(QPoint(x_pos, -self.height - 20))
        self.animation.start()
        QTimer.singleShot(500, self.close)