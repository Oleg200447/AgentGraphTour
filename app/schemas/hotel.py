from pydantic import BaseModel


class HotelBrief(BaseModel):
    hotel_id: int
    teztour_id: int
    name: str
    short_description: str | None = None


class HotelDetails(BaseModel):
    hotel_id: int
    teztour_id: int
    name: str
    description: str | None = None
