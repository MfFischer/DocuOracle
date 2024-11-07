from docuoracle_app import create_app, db
from docuoracle_app.models import User, Document


def init_db():
    app = create_app()
    with app.app_context():
        print("Dropping all tables...")
        db.drop_all()

        print("Creating all tables...")
        db.create_all()

        print("Database initialized successfully!")

        # Optionally create a test user
        try:
            test_user = User(
                username="test_user",
                email="test@example.com"
            )
            test_user.set_password("password123")
            db.session.add(test_user)
            db.session.commit()
            print("Test user created successfully!")
        except Exception as e:
            print(f"Error creating test user: {e}")
            db.session.rollback()


if __name__ == "__main__":
    init_db()